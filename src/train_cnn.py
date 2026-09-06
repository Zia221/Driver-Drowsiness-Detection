import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt


# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32

EPOCHS = 20


# ==========================================
# Load training dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)


# ==========================================
# Load validation dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ==========================================
# Load test dataset
# ==========================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ==========================================
# Display class names
# ==========================================

print("Classes:", train_dataset.class_names)


# ==========================================
# Improve input pipeline
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# Data augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.10),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

])


# ==========================================
# CNN Model
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(128, 128, 3)
    ),

    data_augmentation,

    layers.Rescaling(
        1.0 / 255
    ),


    # Convolution block 1

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convolution block 2

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convolution block 3

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convert feature maps to vector

    layers.Flatten(),


    # Fully connected layer

    layers.Dense(
        128,
        activation="relu"
    ),


    # Reduce overfitting

    layers.Dropout(0.5),


    # Binary classification

    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# ==========================================
# Display model architecture
# ==========================================

model.summary()


# ==========================================
# Compile model
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==========================================
# Train model
# ==========================================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS

)


# ==========================================
# Evaluate test dataset
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ==========================================
# Save model
# ==========================================

model.save(
    "models/eye_state_cnn.keras"
)


print(
    "Model saved successfully!"
)


# ==========================================
# Plot training accuracy
# ==========================================

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.show()


# ==========================================
# Plot training loss
# ==========================================

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.title(
    "CNN Training and Validation Loss"
)

plt.show()