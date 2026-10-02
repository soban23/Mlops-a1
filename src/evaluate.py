import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_data = np.load("data/processed/test.npz")
x_test, y_test = test_data["x"], test_data["y"]

model = tf.keras.models.load_model("models/model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

y_pred = np.argmax(model.predict(x_test), axis=1)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig("confusion_matrix.png")

metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Test accuracy: {test_acc:.4f}")