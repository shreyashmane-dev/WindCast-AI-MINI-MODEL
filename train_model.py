import os
import sys
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
except ImportError:
    EarlyStopping = None
    LSTM = None
    Dense = None
    Dropout = None
    Sequential = None
    load_model = None

if __name__ == "__main__":
    sys.modules["train_model"] = sys.modules[__name__]


DATA_PATH = os.path.join("dataset", "wind.csv")
MODEL_PATH = os.path.join("models", "model.pkl")
FEATURES = ["windspeed", "winddirec", "temperature", "relativehu"]
TARGET = "Power"

MODEL_FILE_NAMES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LSTM": "lstm.keras",
}


class KerasLSTMPredictor:
    def __init__(self, model_path, x_scaler, y_scaler, feature_names, sequence_length):
        self.model_path = model_path
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self._model = None

    def _load_model(self):
        if self._model is None:
            if load_model is None:
                raise RuntimeError(
                    "TensorFlow is required to use the LSTM model. "
                    "Install requirements and retrain the models."
                )
            self._model = load_model(self.model_path)
        return self._model

    def predict(self, x):
        x = pd.DataFrame(x, columns=self.feature_names)
        scaled_x = self.x_scaler.transform(x)
        sequence_x = scaled_x.reshape((scaled_x.shape[0], 1, scaled_x.shape[1]))
        sequence_x = sequence_x.repeat(self.sequence_length, axis=1)
        scaled_prediction = self._load_model().predict(sequence_x, verbose=0)
        return self.y_scaler.inverse_transform(scaled_prediction).ravel()


KerasLSTMPredictor.__module__ = "train_model"


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


def build_lstm_model(feature_count, sequence_length):
    model = Sequential(
        [
            LSTM(
                48,
                input_shape=(sequence_length, feature_count),
                return_sequences=False,
            ),
            Dropout(0.15),
            Dense(24, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def make_sequences(x_values, y_values, sequence_length):
    sequence_x = []
    sequence_y = []
    for index in range(sequence_length, len(x_values)):
        sequence_x.append(x_values[index - sequence_length : index])
        sequence_y.append(y_values[index])

    return sequence_x, sequence_y


def train_lstm_model(data):
    if Sequential is None:
        print("TensorFlow is not installed, skipping LSTM model.")
        return None, None

    os.makedirs("models", exist_ok=True)
    sequence_length = 6
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    scaled_x = x_scaler.fit_transform(data[FEATURES])
    scaled_y = y_scaler.fit_transform(data[TARGET].to_numpy().reshape(-1, 1))
    sequence_x, sequence_y = make_sequences(scaled_x, scaled_y, sequence_length)

    split_index = int(len(sequence_x) * 0.8)
    lstm_x_train = sequence_x[:split_index]
    lstm_x_test = sequence_x[split_index:]
    lstm_y_train = sequence_y[:split_index]
    lstm_y_test = sequence_y[split_index:]

    keras_model = build_lstm_model(len(FEATURES), sequence_length)
    keras_model.fit(
        np.array(lstm_x_train),
        np.array(lstm_y_train),
        epochs=18,
        batch_size=64,
        validation_split=0.15,
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    scaled_predictions = keras_model.predict(np.array(lstm_x_test), verbose=0)
    predictions = y_scaler.inverse_transform(scaled_predictions).ravel()
    y_test = y_scaler.inverse_transform(np.array(lstm_y_test).reshape(-1, 1)).ravel()

    file_path = os.path.join("models", MODEL_FILE_NAMES["LSTM"])
    keras_model.save(file_path)
    predictor = KerasLSTMPredictor(
        file_path, x_scaler, y_scaler, FEATURES, sequence_length
    )

    mae = mean_absolute_error(y_test, predictions)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return predictor, {
        "name": "LSTM",
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
    }


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

    lstm_predictor, lstm_row = train_lstm_model(data)
    if lstm_predictor is not None:
        trained_models["LSTM"] = lstm_predictor
        comparison.append(lstm_row)
        print(
            f"{'LSTM':<22} {lstm_row['mae']:>10.3f} "
            f"{lstm_row['rmse']:>10.3f} {lstm_row['r2']:>10.3f}"
        )
        if lstm_row["rmse"] < best_rmse:
            best_rmse = lstm_row["rmse"]
            best_model = lstm_predictor
            best_name = "LSTM"

    model_files = {}
    for name, trained_model in trained_models.items():
        file_name = MODEL_FILE_NAMES.get(name, f"{name.lower().replace(' ', '_')}.pkl")
        file_path = os.path.join("models", file_name)
        if name != "LSTM":
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
