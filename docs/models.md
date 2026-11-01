# Baselines, Model A and Model B: how they work and why

This explains the third part of the project: the baselines, Model A (pickups per
hexagon-hour) and Model B (minutes until the next pickup). It covers what each model
is, how it is trained, tuned and compared, and why each choice was made. The results
themselves are written by `scripts/evaluate_validation.py` into
`docs/models_validation.md` and `docs/model_card.md`.

How to run it: README section 6c.

## 1. What the two models answer

| | Model A | Model B |
| --- | --- | --- |
| Question | How many real pickups will happen in this hexagon in this hour? | For an empty taxi here, now, how long until its next pickup? |
| One row | Hexagon-hour with at least one empty taxi present | Waiting taxi, every 5 minutes of its wait |
| Label | `target_pickups` (a count: 0, 1, 2, ...) | `minutes_until_end`, censored when `pickup_seen` is False |
| Used in the app for | Demand heat map and the "busy" part of the score | Chance of a pickup within 15 minutes and expected minutes to a pickup |

Model A predicts area demand. Model B predicts the experience of one taxi, which also
depends on competition and on how long it has already been empty. The app needs both.

## 2. Splits: train, validation, test

All splits are by date, never random, because the app is used on future days. A random
split would put the same week (or the same wait) on both sides and make every model look
better than it is.

| Part | Days | Used for |
| --- | --- | --- |
| Warm-up | 1 to 28 January | Left out. History features need 4 weeks before the row. |
| Train | 29 January to 15 September | Fitting and tuning |
| Validation | 16 September to 9 November | Comparing models and freezing the choice |
| Test | 10 November to 31 December | One-time final check in week 3 |
| Excluded | 29 August | Low-coverage day (data problem, see coverage report) |

**Tuning folds.** Tuning never looks at validation. Inside the training period there are
two forward-chaining folds: fit on 29 January to 15 June and score on 16 June to 31 July;
then fit on 29 January to 31 July and score on 1 August to 15 September. Each setting gets
the mean of the two fold scores. Forward-chaining copies how the model will be used (learn
from the past, predict the next weeks), and two folds show whether a setting is good in
more than one period.

**Purging.** A Model B row at 23:50 on 15 September with a 40-minute wait has an outcome
that ends on 16 September, inside validation. Such rows are dropped from train and
validation so no label crosses a split boundary. The test period keeps all its rows,
because it is the last split and nothing comes after it.

**Whole waits stay together.** All snapshots of a wait are stored with the day the wait
started, so one wait is never split between train and validation.

## 3. Sampling

The full tables have about 12.7 million Model A rows and 30 million Model B rows. Training
on all of them is slow and adds little, because neighbouring rows are very similar
(snapshots of one wait 5 minutes apart).

| Table | Train | Validation | Test | Unit sampled |
| --- | --- | --- | --- | --- |
| Model A | 25% | 50% | 50% | hexagon-hour |
| Model B | 12% | 20% | 20% | whole wait (all its snapshots) |

Sampling uses a hash of the key (`episode_id` for Model B, hexagon and hour for Model A),
so the same rows are chosen every time, on every computer, with no random seed to keep.
Model B samples whole waits rather than single snapshots, so a wait is either fully in or
fully out, and the snapshots of one wait cannot leak information across the sample.

## 4. The baselines (what a model has to beat)

The baselines use no machine learning, only averages and counting. They answer the
question every learned model must pass: does it beat what you could get from a table of
past averages? If not, the simpler table is used.

### 4.1 Model A baseline: historical average with shrinkage

For each hexagon, hour of day and weekday, the mean pickups per hour in the training
period. Many such groups are small (a quiet hexagon at 03:00 on Sundays may have only a
few training hours), and a raw mean over 2 hours is noise. So each level is pulled
towards the next coarser one:

```
estimate = (sum of pickups in the group + K x coarser estimate) / (rows in the group + K),   K = 5
```

The chain is city mean, then hexagon, then hexagon x hour, then hexagon x hour x weekday.
A group with 5 rows is half its own mean and half the coarser estimate; a group with 500
rows is almost entirely its own mean; an unseen hexagon gets the city mean.

**Why shrinkage:** in the first smoke run, the plain average gave a Poisson deviance of 3.4
because small groups with zero pickups predicted exactly 0, and Poisson deviance punishes
a prediction of 0 very hard when a pickup then happens. Shrinkage fixed this and makes the
baseline a fair opponent, not a straw man.

### 4.2 Model B baseline: historical survival curves

A Kaplan-Meier curve for each group of area (a square cell of 0.025 degrees, about 2.8 km),
3-hour band of the day and minutes-empty band (0-10, 10-20, 20-40, 40+). A group needs at
least 300 training snapshots; otherwise it falls back to the 3-hour band x minutes band,
then the minutes band, then the whole city.

Kaplan-Meier is the standard way to estimate "chance of still waiting after t minutes"
when some waits are censored: a censored wait counts as "still waiting" up to the time it
was cut and then leaves the count, instead of being thrown away or treated as a pickup.

**Why these groups:** area, time of day and how long the taxi has been empty were the three
biggest differences in the EDA. Cells larger than one hexagon are used because one hexagon
x 3 hours x minutes band is usually far below 300 snapshots.

## 5. Model A: two candidates

Both use the same 25 features (`docs/features.md`) and predict the expected count.

### 5.1 Poisson regression

`sklearn.linear_model.PoissonRegressor`: log(expected pickups) = a weighted sum of
features. It is the textbook model for counts, it is fast and its weights are easy to read.

Because it is linear, the features are prepared first (`src/taxi/models/prep.py`):

- counts (pickups, empty taxis, numbers of waits) get log(1 + x), so 0 vs 5 matters more
  than 100 vs 105;
- hour of day becomes sine and cosine at one and two cycles per day, so 23:00 and 00:00 are
  close and a morning and evening peak can both be fitted; month becomes sine and cosine;
- day of week becomes one column per day (it has no natural order);
- features that can be missing (recent and historical median wait, traffic speed) get a
  "was missing" flag and the training median, so "no data" is not mistaken for a value;
- everything is standardised (mean 0, spread 1), so the penalty treats all weights equally.

The only tuning setting is the L2 penalty `alpha`, tried on a log grid from 0.00001 to 10.

### 5.2 Boosted trees with Poisson loss

`sklearn.ensemble.HistGradientBoostingRegressor(loss="poisson")`, the same algorithm family
as LightGBM. Hundreds of small decision trees are added one at a time, each correcting the
last. Trees find interactions on their own (for example, rain matters at rush hour but not
at 03:00) and need no scaling or logs. Missing values are handled natively.

Settings searched (random search, 10 different settings): learning rate, maximum leaves
per tree, minimum rows per leaf, L2 penalty, share of features per split. The number of
trees is set by early stopping on 10% of the fitting rows (stop after 20 rounds with no
improvement, at most 500).

## 6. Model B: two candidates

Model B rows have one extra feature, `minutes_empty` (how long the taxi has been waiting so
far). Both candidates output a whole survival curve per row: the chance of still waiting at
every minute from 0 to 180. From the curve the app reads:

| Summary | Meaning |
| --- | --- |
| `p_pickup_15` | Chance of a pickup within 15 minutes (1 minus the curve at 15) |
| `median` | Minute at which the curve drops below 0.5 |
| `rmean_60`, `rmean_120` | Average minutes until pickup, counting at most 60 (or 120) minutes |

The restricted means are used for the "expected wait" because a plain mean is not defined
when some waits never end inside the data.

### 6.1 Cox proportional hazards

The classic survival regression: the pickup rate at minute t for a row is a shared
baseline rate times exp(weighted sum of features). It handles censoring directly and its
weights read as "this feature multiplies the pickup rate by x". It is implemented in
`src/taxi/models/survival.py` (no extra library needed): the partial likelihood is fitted
by L-BFGS with an L2 penalty, and the baseline is the Breslow estimate on a 1-minute grid.
It uses the same prepared features as the Poisson regression. Tuning setting: L2 penalty,
log grid from 0.0001 to 10.

**Its limit:** it assumes a feature has the same effect at every minute of the wait, and it
only sees interactions that are written in by hand.

### 6.2 Boosted trees turned into a survival curve

Boosted trees do not handle censoring natively in scikit-learn. So the model is built in
two steps (`ResidualCurveModel`):

1. **Typical wait:** boosted trees predict log(1 + minutes until pickup), trained on
   completed waits (with a pickup) from 90% of the training waits.
2. **Spread around it:** the other 10% of waits (chosen by a hash of the wait, all its
   snapshots together) are held out. For each held-out snapshot the error
   log(1 + actual minutes) minus the predicted value is computed, and a Kaplan-Meier curve
   of these errors is built, **including the censored held-out waits** (for them the error
   is "at least this big").

A row's curve is then "chance of still waiting at minute t" = chance that the error is
bigger than log(1 + t) minus the row's prediction.

**Why step 2 counts censored waits:** step 1 only learns from completed waits, which are
on average shorter than all waits (the long ones are more often cut). In the smoke run,
building the spread from completed waits only made the model too optimistic: 52% predicted
chance of a pickup within 15 minutes against 43% observed. Including the censored waits in
the error curve corrects the level; after the fix the mean prediction is close to observed
(see the calibration figure in the validation report).

Tuning settings: the same random search as Model A, 10 settings.

## 7. Tuning: the same budget for everyone

Every learned model gets **10 settings**, each scored on the same two folds, with the same
cap of 600,000 fitting rows per fold. The best setting is refitted on the whole training
sample. Model A settings are ranked by Poisson deviance (lower is better), Model B by
C-index (higher is better). Every trial is saved in `pipeline_output/models/tuning.json`.

**Why equal budgets:** if the boosted trees got 100 trials and the linear models 3, a win
for the trees could come from the extra search, not from the model. Equal budgets keep the
comparison fair, which is the point of having simple candidates at all.

## 8. Metrics

### Model A

| Metric | What it measures | Why |
| --- | --- | --- |
| Poisson deviance (main) | How likely the actual counts are under the predicted means | The right score for counts; it judges the predicted mean, which the app uses |
| MAE, RMSE | Average size of the miss | Easy to explain |
| MAE on zero hours and non-zero hours | Misses on quiet vs busy hours | 76% of hexagon-hours have zero pickups; this shows whether a model only gets the zeros right |
| Mean predicted vs mean actual | Overall level | A model can rank well but be too high or low |
| Calibration plot | Mean actual against mean predicted in 10 groups of predictions | Shows whether "predicts 1.0" really means about 1 pickup |

Note: MAE favours predicting the median, which is 0 for most hexagon-hours, while the app
needs the mean. So a model can have a lower deviance but a slightly higher MAE than the
historical average. Deviance decides.

### Model B

| Metric | What it measures | Why |
| --- | --- | --- |
| C-index (main) | Of two taxis, how often the model says the right one waits longer (0.5 = coin flip, 1 = perfect); censored pairs are only used when the order is known | The app ranks areas, so ranking is what matters most |
| Brier score at 15 minutes (IPCW) | Squared error of the predicted chance of a pickup within 15 minutes; censored rows are reweighted by the chance of not being censored | Checks the actual probability the app shows, not only the order |
| Calibration at 15 minutes | Predicted chance vs Kaplan-Meier observed chance in 10 groups | Shows whether "60% chance" means about 60% |
| MAE on completed waits | Average miss of the predicted median, completed waits only | Easy to explain, but biased to short waits, so not used to choose |

The C-index is computed on a fixed random subsample of 20,000 rows (it compares pairs, so
the full set would take too long); the same subsample is used for every model.

## 9. Freezing the choice

`evaluate_validation.py` applies a rule written in advance:

- **Model A:** the candidate (Poisson regression or boosted trees) with the lowest
  validation Poisson deviance.
- **Model B:** the candidate (Cox or boosted trees) with the highest validation C-index;
  if both have the same C-index to 3 decimals, the lower Brier score wins.
- The baselines are always kept: the historical survival curves are the fallback for areas
  with too little data, and both baselines are reported next to the frozen models.

The choice, the features and the tuned settings are written to
`pipeline_output/models/final.json`. After that, nothing about the models changes before
the test run. Choosing the rule before seeing the results stops us from picking whichever
metric happens to favour the model we like.

## 10. Extra checks in the validation report

- **E3, robustness:** the Model B C-index in groups where censoring and data differ: quiet
  hours (01:00-05:00), rush hours, central (8 km or less from the centre) and outer areas,
  taxis empty under 15 minutes and 30 minutes or more. A model that only wins in one group
  is a weaker result than one that wins everywhere.
- **E4, feature groups:** boosted trees with the tuned settings, trained on 400,000 rows,
  adding one feature group at a time (calendar and place first, then recent demand,
  competition, traffic, recent waits, history, weather). This shows which data is worth
  having. It uses fixed settings and a smaller sample, so small differences (under about
  0.005) are noise.
- **Permutation importance:** how much the score worsens when one feature's values are
  shuffled in the validation rows. It shows what the model relies on, not what causes
  pickups.

## 11. Validation results (H3 build, 1 Nov 2026)

Run on the H3 build with the default samples: Model A 2,015,085 training and 967,580
validation hexagon-hours; Model B 2,280,472 training and 915,178 validation snapshots
(229 training days, 55 validation days, 52 test days held back). Full tables, figures and
the tuning log are in `docs/models_validation.md`; the frozen choice is in
`pipeline_output/models/final.json` and `docs/model_card.md`.

| Validation | Baseline | Linear candidate | Boosted trees |
| --- | --- | --- | --- |
| Model A Poisson deviance (lower is better) | 0.782 | 0.791 (Poisson regression) | **0.757** |
| Model B C-index (higher is better) | 0.574 | 0.588 (Cox) | **0.604** |
| Model B Brier at 15 min (lower is better) | 0.235 | 0.237 | **0.230** |
| Model B calibration error at 15 min | 0.012 | 0.020 | **0.011** |

**Frozen by the rule:** Model A = boosted trees (Poisson); Model B = boosted-tree curve;
fallback = historical survival curves.

What the results say:

- **Both boosted-tree models beat their baselines, by modest margins.** Model A is 3.1%
  lower in deviance than the historical average, and better in both central (1.085 vs
  1.090) and outer areas (0.704 vs 0.731). Most of the signal is already in "where and
  when"; the app's value comes as much from combining demand, competition and driving
  time as from the model margin.
- **The Poisson regression does not beat the historical average** (0.791 vs 0.782): a
  straight-line formula cannot learn a separate level for each place and hour the way a
  lookup table can. MAE is slightly higher for the boosted trees than for the average
  (0.418 vs 0.408) because MAE rewards predicting the median (0); deviance, which judges the
  expected count the app uses, decides.
- **Model B:** the boosted-tree curve has the best C-index in every E3 group. The margin
  is largest in outer areas (0.594 vs 0.556 for the baseline) and quiet hours (0.597 vs
  0.556), where censoring is highest. Cox nearly ties it in central areas (0.600 vs 0.604)
  and for taxis empty 30+ minutes (0.593 vs 0.595). The trees' mean predicted chance of a
  pickup within 15 minutes is 44.1% against 43.3% observed.
- **A C-index of 0.60 is modest.** Waiting for a street hail is very random; the model
  ranks better than chance and better than history, but individual waits stay hard to
  predict. This is why the app shows a score and a range, not a promise.
- **Most used features:** Model A relies on the 4-week pickups of the hexagon, then its
  location, then empty taxis in the last hour. Model B relies on minutes already empty,
  then hour of day, location, empty taxis nearby (last hour and 4-week).
- **E4 for Model A, and the full-size check.** On the 400,000-row E4 sample, calendar and place
  alone (0.767) scored better than all 25 features (0.772). `scripts/check_model_a_features.py`
  repeated the test on the full training sample (`docs/models_feature_check.md`) and confirmed it:
  calendar and place alone score **0.750**, all 25 features 0.757. Every recent-data group added
  on top makes Model A worse; leaving one group out of the 25 changes little (the groups overlap),
  and only the 4-week history clearly helps. For Model B, every E4 step is within 0.003, so all
  features stay.

**Decision (1 Nov 2026):** Model A stays frozen as it is (boosted trees, all 25 features) for now.
The recommendation, explained in section 12 below, is to test a 7-feature Model A properly before
the test run.

The development run on the earlier stand-in build (smaller samples) gave the same picture
with similar numbers (Model A 0.741 vs 0.763; Model B 0.602 vs 0.566), so the result does not
depend on the grid.

## 12. Recommendation: a simpler Model A (not applied yet)

**What.** Train Model A on the 7 calendar-and-place features only (`hour_of_day`, `day_of_week`,
`is_weekend`, `is_holiday`, `month`, `hex_lat`, `hex_lon`), and also try those 7 plus the 5
history features (12 features). Keep whichever has the lowest validation deviance.

**Why.** On validation the 7-feature model scores 0.750 against 0.757 for the frozen 25-feature
model (more than twice the noise level), and 4.1% better than the historical average instead of
3.1%. Likely reasons, to be checked: one-hour counts per hexagon are mostly 0 or 1, so they carry
a lot of random noise that the trees partly learn; early stopping holds out random rows from the
same hours as training, so it cannot see that this noise does not carry over to later months; and
4-week averages help because averaging removes most of that noise.

**How it improves the project.** Lower validation deviance; a simpler model whose inputs are all
known in advance, so the heat map and demand ranking can be computed for any future hour without
live data; nothing to break when recent data is missing; faster to compute in the replay and API.
Model B is not affected (it does use recent and competition features, and it does not take
Model A as an input).

**How to do it fairly.** Give the 7- and 12-feature versions the same tuning budget as before
(10 settings, same folds), compare on validation only, then freeze and re-run
`evaluate_validation.py` so `final.json` and the reports match. Optionally also test time-based
early stopping (hold out the last weeks of training instead of random rows). All of this happens
before the one-time test run.

## 13. Test run (week 3)

`python scripts/evaluate_test.py --confirm` scores the frozen models and baselines on the
test period once and writes `docs/models_test.md`. It refuses to run a second time (a lock
file is written); `--force` exists but any rerun must be reported, because looking at test
results and then changing the model turns the test set into a second validation set.
