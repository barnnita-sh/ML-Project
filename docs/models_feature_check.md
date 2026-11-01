# Model A feature-group check (full training sample)

Boosted trees (Poisson) with the frozen settings, 2,015,085 training rows, scored on 967,580 validation hexagon-hours. Poisson deviance, lower is better. Differences under about 0.003 are noise.

## Adding one group at a time

| Added group | Features | Deviance |
| --- | --- | --- |
| calendar place | 7 | 0.7499 |
| recent demand | 12 | 0.7592 |
| competition | 16 | 0.7610 |
| traffic | 17 | 0.7607 |
| recent waits | 19 | 0.7616 |
| history | 24 | 0.7569 |
| weather | 25 | 0.7568 |

## Leaving one group out (all others kept)

Positive change = the model is worse without the group, so the group helps.

| Left out | Features | Deviance | Change vs all features |
| --- | --- | --- | --- |
| recent demand | 20 | 0.7568 | +0.0000 |
| competition | 21 | 0.7581 | +0.0014 |
| traffic | 24 | 0.7566 | -0.0002 |
| recent waits | 23 | 0.7565 | -0.0002 |
| history | 20 | 0.7617 | +0.0050 |
| weather | 24 | 0.7569 | +0.0001 |
