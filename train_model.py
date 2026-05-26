import os
from math import sqrt

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DATA_PATH = os.path.join("dataset", "wind.csv")
MODEL_PATH = os.path.join("models", "model.pkl")
FEATURES = ["windspeed", "winddirec", "temperature", "relativehu"]
TARGET = "Power"

MODEL_FILE_NAMES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing dataset/wind.csv")

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    return data[required_columns].dropna()


def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    else:
        print("XGBoost is not installed, skipping XGBoost model.")

    return models


def train_and_select_model():
    data = load_data()
    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    comparison = []
    best_model = None
    best_name = ""
    best_rmse = float("inf")
    trained_models = {}

    print("\nTraining wind power forecasting models...\n")
    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R2 Score':>10}")
    print("-" * 56)

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        predictions = model.predict(x_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        row = {
            "name": name,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
        }
        comparison.append(row)

        print(f"{name:<22} {mae:>10.3f} {rmse:>10.3f} {r2:>10.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    os.makedirs("models", exist_ok=True)

    model_files = {}
    for name, trained_model in trained_models.items():
        file_name = MODEL_FILE_NAMES.get(name, f"{name.lower().replace(' ', '_')}.pkl")
        file_path = os.path.join("models", file_name)
        joblib.dump(trained_model, file_path)
        model_files[name] = file_path

    model_package = {
        "model": best_model,
        "models": trained_models,
        "model_files": model_files,
        "model_name": best_name,
        "features": FEATURES,
        "comparison": comparison,
    }
    joblib.dump(model_package, MODEL_PATH)

    print("-" * 56)
    print(f"Best model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}\n")
    print("Saved individual model files:")
    for name, file_path in model_files.items():
        print(f"- {name}: {file_path}")
    print()

    return model_package


if __name__ == "__main__":
    train_and_select_model()
