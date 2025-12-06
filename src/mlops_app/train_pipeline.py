import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)


def train_model():
    config = load_config()
    set_seed(config["train"]["random_state"])

    logging.info("Loading data...")
    df = pd.read_csv("data/raw/sample.csv")  # you’ll replace this soon

    target_col = config["train"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"],
    )

    logging.info("Training model...")
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logging.info(f"Model accuracy: {acc:.4f}")

    # save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    logging.info("Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()
