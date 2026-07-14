temperature
humidity
predicted_power
created_at
```

## Graphs

The dashboard shows:

```text
Latest predicted power
Average predicted power
Highest predicted power
Lowest predicted power
Power trend graph
Prediction bar graph
Latest weather input gauges
```

The graphs use the latest 12 saved predictions from `windcast.db`.

## Notes

- Keep `dataset/wind.csv` in the same format if replacing the dataset.
- Delete `models/model.pkl` if you want `python app.py` to automatically retrain.
- Delete `windcast.db` if you want to clear previous prediction history.
- This project is designed for local demos and beginner-friendly explanation.
- Updated implementation notes. (2026-07-14 07:46:21.746918)
- Minor documentation improvements. (2026-07-14 07:46:22.561291)
