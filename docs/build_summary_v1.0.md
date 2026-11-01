# Build summary, rules v1.0

Days built: 365 (20250101 to 20251231). Grid: h3.

| Item | Value |
| --- | --- |
| Taxis per day (median) | 1,385 |
| GPS records after cleaning | 369,803,460 |
| Speed jumps removed | 13,588 |
| Light flicker fixed | 762,570 |
| Trips, all | 5,348,692 |
| Trips marked fake (resting with light 'occupied') | 388,623 (7.3%) |
| Real trips | 4,960,069 (median 21.0 min, 5.4 km) |
| Empty periods | 5,646,115 |
| ... starting at a drop-off / after a rest | 5,228,579 / 417,536 |
| ... flagged possible missed trip (left out of snapshots) | 120,681 (2.1%) |
| ... censored | 15.6% |
| Wait until pickup, completed and not flagged (25% / median / 75% / 90%) | 10.0 / 18.0 / 30.7 / 48.0 min |
| Model B snapshot rows | 29,996,060 |

## Why periods ended

| Reason | Periods | Share |
| --- | --- | --- |
| none (pickup seen) | 4,763,384 | 84.4% |
| off_duty | 780,080 | 13.8% |
| gap | 57,489 | 1.0% |
| too_long | 38,345 | 0.7% |
| lookahead_end | 6,591 | 0.1% |
| end_of_data | 226 | 0.0% |

Reasons: `off_duty` = a rest (45+ min parked, engine off) or a fake pickup cut the period;
`too_long` = censored at 180 min; `gap` = GPS silence over 10 min;
`lookahead_end` / `end_of_data` = data ran out.

Censoring is much higher than the 2% in the original EDA because rests and fake pickups
are no longer counted as waits that ended in a passenger. Survival models handle censored rows.
