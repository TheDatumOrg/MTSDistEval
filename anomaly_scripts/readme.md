# MTS on Anomaly Detection Experiments

## Run Experiments

```
# enter into the root folder of this project
bash anomaly_scripts/run_anomaly_exp.sh $YOUR_DATA_PATH
```
Example:
```
# enter into the root folder of this project
bash anomaly_scripts/run_anomaly_exp.sh data/TSB-AD-M
```

## Prepare and analyze results

Step 1: Aggregate all csv files:
```
python anomaly_scripts/aggregate_results.py -d $YOUR_RESULT_PATH -o $YOUR_OUTPUT_PATH
```

Example
```
python anomaly_scripts/aggregate_results.py -d AD_results -o AD_results
```

Step 2: Result Analysis
```
python anomaly_scripts/result_analysis.py  --csv $YOUR_AGGREGATED_CSV_PATH
```

Example:
```
python anomaly_scripts/result_analysis.py  --csv AD_results/summary_AD_results.csv
```




