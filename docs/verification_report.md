# Verification report

Run on 2026-11-01 17:44. 15 of 15 checks passed.

| Check | Result | Detail |
| --- | --- | --- |
| 365 days built | PASS | 365 days |
| all days built with rules v1.0 and H3 | PASS | rules {'v1.0'}, grid {'h3'} |
| table row counts match the manifests | PASS | all days |
| hexagon-hour table has one row per hexagon and hour | PASS | all days |
| hexagon-hour pickups = real trips of the day | PASS | all days |
| Model B feature table has one row per snapshot | PASS | all days |
| Model A table = hexagon-hours with an empty taxi | PASS | all days |
| Model A target = hexagon-hour pickups | PASS | all days |
| snapshot labels positive, minutes_empty in 5-min steps up to 120 | PASS | all days |
| hour_of_day matches the snapshot time | PASS | all days |
| all 25 features present in both tables | PASS | all days |
| hexagon IDs are valid H3 resolution 8 | PASS | example 8864a4b229fffff |
| no missing values in count features (training period) | PASS | max 0.00% |
| rain feature is filled from the weather file | PASS | 4,801,211 snapshot rows had rain > 0.1 mm in the last hour |
| hand re-computation of features for random snapshots | PASS | 320 of 320 matched (days 20250420, 20250812, 20250825, 20250914, 20251016, 20251105, 20251123, 20251206) |

## Missing-value share per feature (training period)

Expected: only the recent and historical wait medians, speed and rain can be missing.

| Feature | Missing |
| --- | --- |
| hour_of_day | 0.00% |
| day_of_week | 0.00% |
| is_weekend | 0.00% |
| is_holiday | 0.00% |
| month | 0.00% |
| hex_lat | 0.00% |
| hex_lon | 0.00% |
| pickups_hex_1h | 0.00% |
| vacant_hex_1h | 0.00% |
| pickups_nbr_1h | 0.00% |
| vacant_nbr_1h | 0.00% |
| vacant_per_pickup_nbr_1h | 0.00% |
| speed_nbr_1h | 0.95% |
| pickups_nbr_3h | 0.00% |
| vacant_nbr_3h_mean | 0.00% |
| pickups_nbr_lag24 | 0.00% |
| pickups_nbr_lag168 | 0.00% |
| hist_pickups_hex_4w | 0.00% |
| hist_pickups_nbr_4w | 0.00% |
| hist_vacant_nbr_4w | 0.00% |
| median_wait_nbr_3h | 6.36% |
| n_waits_nbr_3h | 0.00% |
| hist_wait_nbr_4w | 4.22% |
| n_hist_waits_nbr_4w | 0.00% |
| rain_lag_1h | 0.00% |

Model B rows checked: 29,996,060.
