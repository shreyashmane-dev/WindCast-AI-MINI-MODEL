import os
import sqlite3
from datetime import datetime, timedelta

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for

from train_model import MODEL_PATH, train_and_select_model


app = Flask(__name__)
DB_PATH = "windcast.db"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("models/model.pkl not found. Starting automatic training...")
        return train_and_select_model()

    print("Loading existing model from models/model.pkl")
    package = joblib.load(MODEL_PATH)

    if "models" not in package or "model_files" not in package:
        print("Old model file found. Retraining to save all selectable model files...")
        return train_and_select_model()

    return package


def init_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                windspeed REAL NOT NULL,
                winddirec REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                predicted_power REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(
    windspeed, winddirec, temperature, humidity, predicted_power, created_at=None
):
    if created_at is None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions
            (windspeed, winddirec, temperature, humidity, predicted_power, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                windspeed,
                winddirec,
                temperature,
                humidity,
                predicted_power,
                created_at,
            ),
        )


def get_predictions(limit=12):
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def count_predictions():
    with sqlite3.connect(DB_PATH) as connection:
        total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    return total


def seed_demo_predictions():
    if count_predictions() > 0:
        return

    data = pd.read_csv(os.path.join("dataset", "wind.csv")).dropna().head(12)
    start_time = datetime.now() - timedelta(minutes=55)

    for index, row in data.iterrows():
        input_data = pd.DataFrame(
            [[row["windspeed"], row["winddirec"], row["temperature"], row["relativehu"]]],
            columns=features,
        )
        predicted_power = round(float(model.predict(input_data)[0]), 2)
        created_at = (start_time + timedelta(minutes=int(index) * 5)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        save_prediction(
            float(row["windspeed"]),
            float(row["winddirec"]),
            float(row["temperature"]),
            float(row["relativehu"]),
            predicted_power,
            created_at,
        )


def build_dashboard_data(history):
    trend = list(reversed(history))
    if not trend:
        return {
            "line_points": "",
            "area_points": "",
            "bars": [],
            "gauges": [],
            "latest": 0,
            "average": 0,
            "highest": 0,
            "lowest": 0,
        }

    values = [row["predicted_power"] for row in trend]
    max_value = max(values) if max(values) > 0 else 1
    min_value = min(values)
    width = 720
    height = 260
    padding = 34
    baseline = height - padding

    if len(values) == 1:
        x_step = 0
    else:
        x_step = (width - padding * 2) / (len(values) - 1)

    points = []
    for index, value in enumerate(values):
        x = padding + index * x_step
        y = baseline - (value / max_value) * (height - padding * 2)
        points.append(f"{x:.1f},{y:.1f}")

    area_points = f"{padding},{baseline} {' '.join(points)} {width - padding},{baseline}"

    bar_width = 34 if len(values) <= 12 else 24
    bars = []
    for index, row in enumerate(trend):
        x = padding + index * x_step - bar_width / 2
        bar_height = (row["predicted_power"] / max_value) * (height - padding * 2)
        bars.append(
            {
                "x": round(x, 1),
                "y": round(baseline - bar_height, 1),
                "width": bar_width,
                "height": round(bar_height, 1),
                "value": round(row["predicted_power"], 2),
                "label": f"P{index + 1}",
            }
        )

    latest_row = trend[-1]
    gauge_specs = [
        ("Windspeed", latest_row["windspeed"], 25),
        ("Direction", latest_row["winddirec"], 360),
        ("Temperature", latest_row["temperature"], 45),
        ("Humidity", latest_row["humidity"], 100),
    ]
    gauges = []
    for label, value, maximum in gauge_specs:
        percent = max(0, min(100, (value / maximum) * 100))
        gauges.append(
            {
                "label": label,
                "value": round(value, 2),
                "percent": round(percent, 1),
            }
        )

    return {
        "line_points": " ".join(points),
        "area_points": area_points,
        "bars": bars,
        "gauges": gauges,
        "latest": round(values[-1], 2),
        "average": round(sum(values) / len(values), 2),
        "highest": round(max_value, 2),
        "lowest": round(min_value, 2),
    }


model_package = ensure_model()
models = model_package["models"]
model = model_package["model"]
features = model_package["features"]
model_name = model_package["model_name"]
model_comparison = model_package["comparison"]
model_choices = list(models.keys())
init_database()
seed_demo_predictions()


@app.route("/")
def index():
    history = get_predictions()
    dashboard = build_dashboard_data(history)

    return render_template(
        "index.html",
        prediction=None,
        latest_inputs=None,
        history=history,
        dashboard=dashboard,
        model_name=model_name,
        selected_model=model_name,
        model_choices=model_choices,
        model_comparison=model_comparison,
    )


@app.route("/predict", methods=["POST"])
def predict():
    windspeed = float(request.form["windspeed"])
    winddirec = float(request.form["winddirec"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    selected_model = request.form.get("model_choice", model_name)
    selected_predictor = models.get(selected_model, model)

    input_data = pd.DataFrame(
        [[windspeed, winddirec, temperature, humidity]],
        columns=features,
    )
    predicted_power = round(float(selected_predictor.predict(input_data)[0]), 2)

    save_prediction(windspeed, winddirec, temperature, humidity, predicted_power)
    history = get_predictions()
    dashboard = build_dashboard_data(history)

    latest_inputs = {
        "windspeed": windspeed,
        "winddirec": winddirec,
        "temperature": temperature,
        "humidity": humidity,
    }

    return render_template(
        "index.html",
        prediction=predicted_power,
        latest_inputs=latest_inputs,
        history=history,
        dashboard=dashboard,
        model_name=model_name,
        selected_model=selected_model,
        model_choices=model_choices,
        model_comparison=model_comparison,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    windspeed = float(request.form["windspeed"])
    winddirec = float(request.form["winddirec"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    selected_model = request.form.get("model_choice", model_name)
    selected_predictor = models.get(selected_model, model)

    input_data = pd.DataFrame(
        [[windspeed, winddirec, temperature, humidity]],
        columns=features,
    )
    predicted_power = round(float(selected_predictor.predict(input_data)[0]), 2)

    return jsonify(
        {
            "prediction": predicted_power,
            "selected_model": selected_model,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
