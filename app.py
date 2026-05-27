import csv
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from io import StringIO

import joblib
import pandas as pd
from flask import Flask, jsonify, make_response, render_template, request, url_for

from train_model import MODEL_PATH, train_and_select_model


app = Flask(__name__)
DB_PATH = "windcast.db"
MODEL_DETAILS = {
    "Linear Regression": {
        "summary": "Fast baseline model that fits a straight-line relationship between weather inputs and power.",
        "strength": "Easy to understand and useful as a benchmark.",
    },
    "Decision Tree": {
        "summary": "Rule-based regressor that splits wind and weather ranges into smaller decisions.",
        "strength": "Captures simple non-linear patterns with clear logic.",
    },
    "Random Forest": {
        "summary": "An ensemble of decision trees that averages many forecasts for a steadier result.",
        "strength": "Strong general-purpose accuracy and good resistance to noisy readings.",
    },
    "XGBoost": {
        "summary": "Gradient-boosted trees that learn from previous errors to improve each stage.",
        "strength": "Often performs well on structured forecasting datasets.",
    },
    "LSTM": {
        "summary": "A neural sequence model built with Keras to learn temporal wind-power patterns.",
        "strength": "Designed for time-series behavior and recent-condition memory.",
    },
}


def get_lstm_status():
    installed = False
    try:
        import tensorflow  # noqa: F401

        installed = True
    except ImportError:
        installed = False

    return {
        "installed": installed,
        "trained": "LSTM" in models if "models" in globals() else False,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "recommended_python": "3.10 or 3.11",
    }


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("models/model.pkl not found. Starting automatic training...")
        return train_and_select_model()

    print("Loading existing model from models/model.pkl")
    package = joblib.load(MODEL_PATH)

    if "models" not in package or "model_files" not in package:
        print("Old model file found. Retraining to save all selectable model files...")
        return train_and_select_model()

    if "LSTM" not in package["models"]:
        try:
            import tensorflow  # noqa: F401

            print("LSTM model missing from package. Retraining with TensorFlow...")
            return train_and_select_model()
        except ImportError:
            print("TensorFlow is not installed. Existing non-LSTM models loaded.")

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
        if limit is None:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                ORDER BY id DESC
                """
            ).fetchall()
        else:
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


def build_wind_rose_data(history):
    bins = [0] * 8
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    if not history:
        return []
        
    for row in history:
        deg = row["winddirec"] % 360
        bin_idx = int(((deg + 22.5) % 360) // 45)
        if 0 <= bin_idx < 8:
            bins[bin_idx] += 1
            
    total = len(history)
    max_count = max(bins) if max(bins) > 0 else 1
    
    rose_data = []
    cx, cy = 100, 100
    for idx in range(8):
        count = bins[idx]
        pct = count / total
        r = (count / max_count) * 80 if max_count > 0 else 0
        if r < 10:
            r = 10
            
        angle_deg = idx * 45 - 90
        a1 = math.radians(angle_deg - 20)
        a2 = math.radians(angle_deg + 20)
        
        x1 = cx + r * math.cos(a1)
        y1 = cy + r * math.sin(a1)
        x2 = cx + r * math.cos(a2)
        y2 = cy + r * math.sin(a2)
        
        path = f"M {cx} {cy} L {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 0 1 {x2:.1f} {y2:.1f} Z"
        
        rose_data.append({
            "label": labels[idx],
            "count": count,
            "pct": round(pct * 100, 1),
            "path": path,
            "lx": round(cx + 92 * math.cos(math.radians(angle_deg)), 1),
            "ly": round(cy + 92 * math.sin(math.radians(angle_deg)) + 4, 1)
        })
        
    return rose_data


def build_power_curve_data(history):
    if not history:
        return {"dots": [], "curve": ""}
        
    w_min, w_max = 0.0, 20.0
    p_min, p_max = 0.0, 1.0
    
    cx_min, cx_max = 40.0, 340.0
    cy_min, cy_max = 170.0, 20.0
    
    dots = []
    for row in history:
        ws = row["windspeed"]
        power = row["predicted_power"]
        
        ws_clamped = max(w_min, min(w_max, ws))
        power_clamped = max(p_min, min(p_max, power))
        
        x = cx_min + ((ws_clamped - w_min) / (w_max - w_min)) * (cx_max - cx_min)
        y = cy_min - ((power_clamped - p_min) / (p_max - p_min)) * (cy_min - cy_max)
        
        dots.append({
            "x": round(x, 1),
            "y": round(y, 1),
            "ws": round(ws, 2),
            "power": round(power, 2)
        })
        
    curve_points = []
    for step in range(41):
        ws = step * 0.5
        if ws < 3.0:
            power = 0.0
        elif ws > 18.0:
            power = 0.0
        else:
            power = 1.0 / (1.0 + math.exp(-(ws - 7.5) / 1.8))
            
        x = cx_min + ((ws - w_min) / (w_max - w_min)) * (cx_max - cx_min)
        y = cy_min - ((power - p_min) / (p_max - p_min)) * (cy_min - cy_max)
        curve_points.append(f"{x:.1f},{y:.1f}")
        
    curve_path = " ".join(curve_points)
    
    return {
        "dots": dots,
        "curve": curve_path
    }


def build_dashboard_data(history):
    history_trend = history[:12]
    trend = list(reversed(history_trend))
    wind_rose = build_wind_rose_data(history)
    power_curve = build_power_curve_data(history)

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
            "wind_rose": [],
            "power_curve": {"dots": [], "curve": ""}
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
        "wind_rose": wind_rose,
        "power_curve": power_curve,
    }


def build_model_cards():
    cards = []
    comparison_by_name = {row["name"]: row for row in model_comparison}
    names = list(MODEL_DETAILS.keys())
    for choice in model_choices:
        if choice not in names:
            names.append(choice)

    for choice in names:
        cards.append(
            {
                "name": choice,
                "metric": comparison_by_name.get(choice),
                "available": choice in model_choices,
                "status": "Selectable" if choice in model_choices else "Needs setup",
                "summary": MODEL_DETAILS.get(choice, {}).get(
                    "summary", "Selectable forecasting model."
                ),
                "strength": MODEL_DETAILS.get(choice, {}).get(
                    "strength", "Compares against the same test split."
                ),
            }
        )

    return cards


# Detect tensorflow availability
has_tensorflow = False
try:
    import tensorflow  # noqa: F401
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

model_package = ensure_model()
models = model_package["models"]

# Filter out LSTM if TensorFlow is not available in the current runtime environment
if not has_tensorflow and "LSTM" in models:
    del models["LSTM"]

model = model_package["model"]
model_name = model_package["model_name"]

if model_name == "LSTM" and not has_tensorflow:
    # Fallback to another trained model if LSTM was active but tensorflow is missing
    available_choices = [m for m in models.keys() if m != "LSTM"]
    model_name = available_choices[0] if available_choices else "Linear Regression"

features = model_package["features"]
model_comparison = model_package["comparison"]

# Filter out LSTM metrics if tensorflow is missing
if not has_tensorflow:
    model_comparison = [row for row in model_comparison if row["name"] != "LSTM"]

model_choices = list(models.keys())
init_database()
seed_demo_predictions()


@app.route("/")
def index():
    lstm_status = get_lstm_status()

    return render_template(
        "index.html",
        model_name=model_name,
        model_choices=model_choices,
        model_cards=build_model_cards(),
        has_lstm="LSTM" in models,
        lstm_status=lstm_status,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    latest_inputs = None
    selected_model = model_name
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M")

    if request.method == "POST":
        windspeed = float(request.form["windspeed"])
        winddirec = float(request.form["winddirec"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        selected_model = request.form.get("model_choice", model_name)
        selected_predictor = models.get(selected_model, model)

        # Parse prediction timestamp
        prediction_time = request.form.get("prediction_time")
        created_at = None
        if prediction_time:
            try:
                dt = datetime.strptime(prediction_time, "%Y-%m-%dT%H:%M")
                created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        input_data = pd.DataFrame(
            [[windspeed, winddirec, temperature, humidity]],
            columns=features,
        )
        prediction = round(float(selected_predictor.predict(input_data)[0]), 2)

        save_prediction(windspeed, winddirec, temperature, humidity, prediction, created_at=created_at)
        latest_inputs = {
            "windspeed": windspeed,
            "winddirec": winddirec,
            "temperature": temperature,
            "humidity": humidity,
            "prediction_time": prediction_time,
        }

    history = get_predictions(limit=36)
    dashboard = build_dashboard_data(history)

    return render_template(
        "predict.html",
        prediction=prediction,
        latest_inputs=latest_inputs,
        dashboard=dashboard,
        model_name=model_name,
        selected_model=selected_model,
        model_choices=model_choices,
        model_comparison=model_comparison,
        current_time=current_time,
    )


@app.route("/history")
def history_page():
    history = get_predictions(limit=None)
    return render_template("history.html", history=history)


@app.route("/about")
def about_page():
    lstm_status = get_lstm_status()
    return render_template(
        "about.html",
        lstm_status=lstm_status,
        model_name=model_name,
        model_comparison=model_comparison,
        selected_model=model_name,
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


@app.route("/download-report")
def download_report():
    history = get_predictions(limit=None)
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Wind Speed (m/s)", "Wind Direction (deg)", "Temperature (F)", "Humidity (%)", "Predicted Power (MW)", "Timestamp"])
    for row in history:
        cw.writerow([
            row["id"],
            row["windspeed"],
            row["winddirec"],
            row["temperature"],
            row["humidity"],
            row["predicted_power"],
            row["created_at"]
        ])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=windcast_forecast_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output


if __name__ == "__main__":
    app.run(debug=True)
