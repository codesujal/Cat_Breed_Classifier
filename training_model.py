import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


# FORCE CPU (safe mode)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# PATHS
IMG_DIR = "oxford-iiit-pet/images"
LIST_FILE = "oxford-iiit-pet/annotations/list.txt"
IMG_SIZE = (224, 224)


# LOAD DATA (CAT ONLY)
cat_images = []
cat_labels = []

with open(LIST_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("#") or len(line) == 0:
            continue

        parts = line.split()

        img_name = parts[0]
        species = int(parts[2])
        breed_id = int(parts[3])

        if species == 1:  # CAT ONLY
            img_path = os.path.join(IMG_DIR, img_name + ".jpg")

            if os.path.exists(img_path):
                cat_images.append(img_path)
                cat_labels.append(breed_id)

print("Total cat images:", len(cat_images))


# LOAD IMAGES
X = []
y = []

for img_path, label in zip(cat_images, cat_labels):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    X.append(img_array)
    y.append(label)

X = np.array(X)
y = np.array(y)


# LABEL ENCODING
unique_labels = np.unique(y)
label_map = {v: i for i, v in enumerate(unique_labels)}
y = np.array([label_map[i] for i in y])

num_classes = len(unique_labels)
print("Cat breeds:", num_classes)


# TRAIN TEST SPLIT
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# MODEL (TRANSFER LEARNING)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# TRAIN (SAVE HISTORY)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=16
)

#model.save("catbreed_model.h5")


# SAVE MODEL (.keras)
model.save("catbreed_model.keras")
print("Model saved.")


# PLOTS
# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Training complete.")