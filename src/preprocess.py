import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

train_data = np.load("data/raw/train.npz")
test_data = np.load("data/raw/test.npz")

x_train_full = train_data["x"].astype("float32") / 255.0
y_train_full = train_data["y"]
x_test = test_data["x"].astype("float32") / 255.0
y_test = test_data["y"]

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=params["test_size"],
    random_state=params["seed"],
)

np.savez("data/processed/train.npz", x=x_train, y=y_train)
np.savez("data/processed/val.npz", x=x_val, y=y_val)
np.savez("data/processed/test.npz", x=x_test, y=y_test)

print(f"train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}")