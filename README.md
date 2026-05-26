# WindCast AI - Short-Term Wind Power Forecasting

WindCast AI is a very simple Flask machine learning web app that predicts wind power output from manual weather inputs.

The app uses:

- Flask
- Python
- HTML
- CSS
- SQLite
- scikit-learn
- XGBoost
- joblib

No React, Docker, Firebase, Bootstrap, or complex frontend setup is required.

## Features

- Manual wind power prediction form
- Instant ML prediction
- Choose between 4 trained models
- Automatic model training if `models/model.pkl` is missing
- SQLite database for previous predictions
- Prediction history table
- Advanced live graphs for the latest 12 predictions
- Prediction metric cards, trend graph, bar graph, and input gauges
- Model comparison table with MAE, RMSE, and R2 Score
- One-command app startup

## Project Structure

```text
windcast-ai/
|
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- windcast.db
|-- README.md
|
|-- dataset/
|   |-- wind.csv
|
|-- models/
|   |-- model.pkl
|   |-- linear_regression.pkl
|   |-- decision_tree.pkl
|   |-- random_forest.pkl
|   |-- xgboost.pkl
|
|-- templates/
|   |-- index.html
|
|-- static/
    |-- style.css
```

## Dataset

The app trains from:

```text
dataset/wind.csv
```

Required columns:

```text
windspeed
winddirec
temperature
relativehu
Power
```

The current dataset was prepared from:

```text
C:\Users\OMPRASAD\Downloads\Wind Power Generation Data Forecasting\Location1.csv
```

Column mapping used:

```text
windspeed_100m          -> windspeed
winddirection_100m      -> winddirec
temperature_2m          -> temperature
relativehumidity_2m     -> relativehu
Power                   -> Power
```

## How To Run

Open a terminal inside the project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

## Important Startup Flow

When you run:

```bash
python app.py
```

The app does this automatically:

1. Checks if `models/model.pkl` exists.
2. If the model is missing, trains the ML models using `dataset/wind.csv`.
3. Compares Linear Regression, Decision Tree, Random Forest, and XGBoost.
4. Selects the best model based on lowest RMSE.
5. Saves all trained models and the best model name to `models/model.pkl`.
6. Also saves each model as a separate `.pkl` file inside `models/`.
7. Creates `windcast.db` if needed.
8. Starts the Flask web server.

## Train Model Manually

You can also train the model directly:

```bash
python train_model.py
```

This prints model comparison results in the terminal and saves the best model.

Example output:

```text
Model                         MAE       RMSE   R2 Score
--------------------------------------------------------
Linear Regression           0.141      0.179      0.614
Decision Tree               0.127      0.167      0.663
Random Forest               0.124      0.163      0.679
XGBoost                     0.126      0.164      0.673
--------------------------------------------------------
Best model: Random Forest
Saved model to: models/model.pkl
```

## Manual Prediction Inputs

The web form accepts:

```text
windspeed
winddirec
temperature
humidity
```

The model predicts:

```text
Power Output
```

## Model Selection

The prediction form includes a model dropdown.

Available models:

```text
Linear Regression
Decision Tree
Random Forest
XGBoost
```

After entering input values once, changing the dropdown immediately submits the form again and updates the predicted result using the selected model.

The app keeps one combined model package:

```text
models/model.pkl
```

It also saves separate visible model files:

```text
models/linear_regression.pkl
models/decision_tree.pkl
models/random_forest.pkl
models/xgboost.pkl
```

The visible model cards update the prediction immediately without saving duplicate rows to history. The `Predict Power` button saves the prediction to SQLite history.

## Database

The app uses local SQLite only.

Database file:

```text
windcast.db
```

Table:

```text
predictions
```

Stored columns:

```text
id
windspeed
winddirec
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
