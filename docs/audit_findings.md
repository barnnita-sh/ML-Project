# Audit findings and rules v1.0

## What the audit was

The pipeline turns GPS pings into trips and empty periods using automatic rules
(light off = pickup, light on = drop-off). Before building the whole year we checked
those rules against what the GPS traces actually show.

- **Sample:** 50 empty periods from 9 training-period days (one per month, January to
  mid-September 2025), built with the first rules (v0.1). Picked in 7 groups so unusual
  cases were covered: night (8), very short under 3 min (7), long 60 to 180 min (8),
  over 3 hours (7), ended by a GPS gap (6), mostly engine off (7), random (7).
- **Pictures:** one per period (`audit/plots/`): a map in km around the drop-off, plus
  speed, engine and light over time.
- **Labels:** each period labelled as `searching`, `waiting_at_spot`, `off_duty`,
  `light_error_or_missed_trip`, `gps_problem` or `unsure`. Claude labelled all 50 with a
  reason and confidence (`audit/claude_labels.csv`); the team's labels matched exactly.
  Final labels are in `audit/audit_sheet.csv` (`agreed_label`, reason in `notes`).

| Label | Count |
| --- | --- |
| searching | 24 |
| waiting_at_spot | 8 |
| off_duty | 8 |
| light_error_or_missed_trip | 6 |
| unsure | 3 |
| gps_problem | 1 |

So about two thirds of sampled periods were real waiting (searching or waiting at a spot).
Because the sample deliberately over-represents odd cases, the share across all periods is higher.

## What we found, and the rule each finding led to

| # | Finding | Evidence (audit #) | Rule in v1.0 |
| --- | --- | --- | --- |
| 1 | Drivers switch the light to "occupied" when they park to rest. This creates fake pickups that end waits early and fake drop-offs that start them. | 4, 11, 12, 13, 19, 33, 41 | If the taxi stays parked with the engine off for 5+ min right after a "pickup", the wait is censored (`off_duty`), not counted as ending with a passenger. A whole "trip" under 0.5 km that is mostly parked is `is_fake` and the wait after it starts as `after_rest`. |
| 2 | An engine off does not mean off duty. Waits with the engine off up to 39 min ended with a passenger at the same spot; clear rests lasted 54 min to 6.5 h. | waiting: 37, 38, 39, 42, 43; rests: 17, 24, 25, 28, 30, 36, 40 | A stretch parked with the engine off for 45+ min is a rest: the wait is cut there (`off_duty`) and a new wait (`after_rest`) starts when the taxi moves. |
| 3 | The old "over 3 hours = break, delete it" rule dropped real searches and kept rests just under 3 h. | dropped: 26, 27; kept: 17, 40 | Rests are handled by rule 2. Anything still over 180 min is censored at 180 (`too_long`) instead of deleted. |
| 4 | Fast long drives out of the city while "empty" are probably passenger trips without the light. | 22, 31, 35 | `possible_missed_trip` if more than 40 km at over 40 km/h average, or ending more than 30 km away. Kept in the empty table, left out of the snapshot table. Slow cruising over many km (18, 27) is not flagged. |
| 5 | Zero-length periods exist (drop-off right before a GPS silence). | 34 | Dropped. |
| 6 | No problems seen with the 10-min gap rule, the flicker fix or the 150 km/h jump rule. | 1 (9-min gap looked fine) | Unchanged. |

## Check: what v1.0 does to the 50 audited periods

`python scripts/check_audit_rules.py` rebuilds each audited day with the new rules
(`audit/rules_check_v1.0.csv`).

| Agreed label | What v1.0 does |
| --- | --- |
| searching (24) | 19 kept as waits ending in a pickup; 2 kept but censored because they ended with the driver going off duty (23, 41); 2 censored at 180 min (26, 27); 1 flagged as a possible missed trip (7, a 37 km drive at 2 am) |
| waiting_at_spot (8) | 7 kept with their pickup; 1 removed as a rest (21: 74 min engine off, low-confidence label) |
| off_duty (8) | all 8 cut at the rest or censored |
| light_error_or_missed_trip (6) | 2 censored as off duty (4, 11), 2 flagged (31, 35), 1 starts after a rest and ends at a GPS gap (33), 1 censored at 180 min (29) |
| gps_problem (1) | dropped |
| unsure (3) | 1 censored as off duty (19), 1 flagged (22), 1 kept (5) |

Known trade-offs: #21 (a 74-min engine-off wait that ended in a pickup) is now treated as a
rest; #7 (likely repositioning) is flagged. Both were medium or low confidence.

## Effect on the full year

See `docs/build_summary_v1.0.md`. In short: 7.3% of trips are fake, 13.8% of empty periods
now end censored as off duty, 2.1% are flagged as possible missed trips, and the median
completed wait stays at 18 minutes.
