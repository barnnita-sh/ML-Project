# Feature coverage report

Model B snapshot rows: 29,996,060. A feature has 'coverage' when there is real activity behind it; a zero is still a valid value, but a model can learn little from mostly-zero features. Central = within 8 km of Siam.

## Overall (January warm-up excluded)

| Check | All | Central | Outer |
| --- | --- | --- | --- |
| pickup in hexagon, last hour | 44% | 64% | 32% |
| pickup in neighbourhood, last hour | 84% | 99% | 76% |
| pickup in neighbourhood, last 3 hours | 95% | 100% | 92% |
| any completed wait nearby, last 3 hours | 94% | 100% | 90% |
| 5+ completed waits nearby, last 3 hours | 70% | 97% | 54% |
| 4-week pickup history > 0 (neighbourhood) | 96% | 100% | 94% |
| taxi speed known (neighbourhood, last hour) | 99% | 100% | 99% |
| snapshots | 27,908,694 | 10,057,192 | 17,851,502 |

## By period

| Check | warm-up | train | validation | test |
| --- | --- | --- | --- | --- |
| pickup in hexagon, last hour | 40% | 43% | 44% | 43% |
| pickup in neighbourhood, last hour | 82% | 84% | 84% | 84% |
| pickup in neighbourhood, last 3 hours | 94% | 95% | 95% | 95% |
| any completed wait nearby, last 3 hours | 92% | 94% | 94% | 94% |
| 5+ completed waits nearby, last 3 hours | 66% | 70% | 70% | 70% |
| 4-week pickup history > 0 (neighbourhood) | 0% | 96% | 96% | 96% |
| taxi speed known (neighbourhood, last hour) | 99% | 99% | 99% | 99% |

In the warm-up month the 4-week history and some lags are missing by design (NaN).

## By hour of day (January excluded)

| Hour | Snapshots | pickup in hexagon, last hour | pickup in neighbourhood, last hour | pickup in neighbourhood, last 3 hours | 5+ completed waits nearby, last 3 hours |
| --- | --- | --- | --- | --- | --- |
| 00:00 | 524,937 | 39% | 79% | 94% | 72% |
| 01:00 | 409,993 | 40% | 79% | 93% | 67% |
| 02:00 | 368,007 | 36% | 76% | 92% | 62% |
| 03:00 | 353,783 | 33% | 73% | 90% | 56% |
| 04:00 | 448,961 | 28% | 68% | 86% | 46% |
| 05:00 | 679,751 | 27% | 68% | 84% | 42% |
| 06:00 | 962,459 | 30% | 73% | 86% | 43% |
| 07:00 | 1,231,778 | 33% | 76% | 88% | 47% |
| 08:00 | 1,369,463 | 36% | 81% | 92% | 56% |
| 09:00 | 1,399,827 | 41% | 85% | 95% | 64% |
| 10:00 | 1,465,811 | 45% | 86% | 96% | 71% |
| 11:00 | 1,503,144 | 48% | 87% | 96% | 75% |
| 12:00 | 1,537,932 | 50% | 88% | 96% | 77% |
| 13:00 | 1,504,700 | 50% | 88% | 96% | 77% |
| 14:00 | 1,546,790 | 50% | 87% | 96% | 77% |
| 15:00 | 1,637,479 | 48% | 87% | 96% | 76% |
| 16:00 | 1,734,294 | 47% | 87% | 96% | 75% |
| 17:00 | 1,851,146 | 45% | 87% | 96% | 75% |
| 18:00 | 1,771,157 | 45% | 87% | 96% | 75% |
| 19:00 | 1,563,458 | 45% | 87% | 96% | 76% |
| 20:00 | 1,348,517 | 46% | 87% | 97% | 77% |
| 21:00 | 1,121,230 | 47% | 87% | 97% | 77% |
| 22:00 | 894,244 | 46% | 86% | 96% | 78% |
| 23:00 | 679,833 | 42% | 82% | 96% | 75% |

## What this means

- Single-hexagon, single-hour pickup counts are mostly zero; the neighbourhood and 3-hour versions carry most of the signal, as planned.
- Night hours and outer areas have the least coverage; expect weaker predictions there and use the coverage rule (Section 10 of the team document) to fall back or say 'not enough data'.
- Full table by hour: `docs/coverage_by_hour.csv`.

## Model A rows

12,738,845 hexagon-hours with an empty taxi present; 26% have at least one pickup; mean 0.37, max 39 pickups.
