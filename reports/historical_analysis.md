# Historical Visit Evidence Analysis

Exploratory analysis only. `reason_reported` is not used as a predictive feature.

## Coverage

- Telemetry rows: **1,433,387**
- Telemetry dates: **2025-08-01 to 2026-04-01**
- Historical visits: **642**
- Visits with pre-request telemetry in 28 days: **314**
- Visits without pre-request telemetry in 28 days: **328**

## Historical outcome counts

outcome
Kein Fehler gefunden    390
Fehler behoben          223
Kein Zugang              29

## Pre-visit feature comparison

                      feature  fixed_n  no_error_n  fixed_median  no_error_median  median_diff   abs_diff
offline_duration_sec_r28d_sum      116         184    44088.8676        4490.4064   39598.4613 39598.4613
 offline_duration_sec_r7d_sum      116         184    29063.9791        1217.4145   27846.5646 27846.5646
     offline_duration_sec_r7d      116         184     4258.0523         173.9164    4084.1360  4084.1360
    offline_duration_sec_r28d      116         184     2288.6304         180.0390    2108.5914  2108.5914
   disconnection_cnt_r28d_sum      116         184       24.0450           9.0665      14.9785    14.9785
    disconnection_cnt_r7d_sum      116         184       11.4617           2.1119       9.3499     9.3499
        disconnection_cnt_r7d      116         184        1.6435           0.3038       1.3397     1.3397
       disconnection_cnt_r28d      116         184        0.8911           0.3259       0.5652     0.5652
                    read_rate      114         175        0.5392           0.8885      -0.3493     0.3493
          reboot_cnt_r28d_sum      116         184        0.3642           0.0417       0.3226     0.3226
           reboot_cnt_r7d_sum      116         184        0.1955           0.0000       0.1955     0.1955
                 read_rate_4w      114         175        0.7339           0.8967      -0.1628     0.1628
               reboot_cnt_r7d      116         184        0.0279           0.0000       0.0279     0.0279
           ecio_bad_share_r7d      115         184        0.3341           0.3100       0.0241     0.0241
              reboot_cnt_r28d      116         184        0.0130           0.0015       0.0115     0.0115
           rssi_bad_share_r7d      116         184        0.1374           0.1264       0.0110     0.0110
          ecio_bad_share_r28d      115         184        0.3118           0.3147      -0.0029     0.0029
          rssi_bad_share_r28d      116         184        0.1273           0.1245       0.0028     0.0028
           rsrp_bad_share_r7d      115         183        0.0052           0.0079      -0.0027     0.0027
          rsrp_bad_share_r28d      115         183        0.0100           0.0120      -0.0020     0.0020
             crc_bad_rate_r7d      116         184        1.0013           1.0003       0.0011     0.0011
            crc_bad_rate_r28d      116         184        1.0000           0.9996       0.0004     0.0004
             tx_busy_rate_r7d      115         184        0.0000           0.0000       0.0000     0.0000
            tx_busy_rate_r28d      115         184        0.0000           0.0000       0.0000     0.0000

## Interpretation guardrails

- Gateway IDs are normalized before matching across telemetry, field visits, and meter-read data.
- Telemetry features use only dates strictly before `requested_on`.
- 7-day and 28-day windows are calendar windows immediately before the visit date.
- `reason_reported` is descriptive historical information and is not used as a prediction feature.
- `Kein Zugang` is retained as an operational outcome but is not treated as a confirmed gateway fault.
- Outcome differences are evidence for further testing, not proof of causation.
- Final decisions should be evaluated using the challenge's asymmetric visit and missed-fault costs.