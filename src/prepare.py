import numpy as np
import os
import tensorflow as tf

os.makedirs("data/raw", exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

np.savez("data/raw/train.npz", x=x_train, y=y_train)
np.savez("data/raw/test.npz", x=x_test, y=y_test)

print(f"Saved {x_train.shape[0]} train and {x_test.shape[0]} test images to data/raw/")