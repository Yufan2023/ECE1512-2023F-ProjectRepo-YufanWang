#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Specify the path to the 'mhist.zip' file
mhist_zip_path = 'drive/MyDrive/mhist_dataset'

# Load annotations from the CSV file
annotations_path = "drive/MyDrive/mhist_dataset/annotations.csv"
annotations_df = pd.read_csv(annotations_path, delimiter=',')

# Filter and split data based on the 'Partition' column
train_annotations = annotations_df[annotations_df['Partition'] == 'train']
test_annotations = annotations_df[annotations_df['Partition'] == 'test']

# Path to the directory containing the images
images_dir = "drive/MyDrive/mhist_dataset/images"

# Initialize empty lists to store image paths and corresponding labels
train_image_paths = []
train_labels = []
test_image_paths = []
test_labels = []

# Process the training data
for index, row in train_annotations.iterrows():
    image_name = row['Image Name']
    image_path = os.path.join(images_dir, image_name)
    label = row['Majority Vote Label']
    train_image_paths.append(image_path)
    train_labels.append(label)

# Process the testing data
for index, row in test_annotations.iterrows():
    image_name = row['Image Name']
    image_path = os.path.join(images_dir, image_name)
    label = row['Majority Vote Label']
    test_image_paths.append(image_path)
    test_labels.append(label)

# Encode labels as integers
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Function to compute student loss
def compute_student_loss(y_true, y_pred, alpha=0.5, temperature=1.0):
    y_true = tf.cast(y_true, tf.float32)  # Ensure both tensors have the same data type
    y_pred = tf.cast(y_pred, tf.float32)

    soft_student = tf.nn.softmax(y_pred / temperature)
    soft_teacher = tf.nn.softmax(y_true / temperature)

    return alpha * tf.keras.losses.categorical_crossentropy(soft_teacher, soft_student) + (1 - alpha) * tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Function to load and preprocess images, converting to RGB
def load_and_preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)  # Assuming images are grayscale
        image = tf.image.resize(image, (224, 224))  # Resize images to match the expected input size for pre-trained models
        image = tf.image.grayscale_to_rgb(image)  # Convert to RGB (3 channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image /= 255.0  # Normalize to [0, 1]
        images.append(image)
    return np.array(images)

# Load and preprocess images
train_images = load_and_preprocess_images(train_image_paths)
test_images = load_and_preprocess_images(test_image_paths)

# One-hot encode the labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=len(label_encoder.classes_))

# Split the training data into training and validation sets
train_images, val_images, train_labels_encoded, val_labels_encoded, train_labels_one_hot, val_labels_one_hot = train_test_split(
    train_images, train_labels_encoded, train_labels_one_hot, test_size=0.2, random_state=42)

# Define the teacher model
def build_teacher_model():
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    return model

# Define the student model
def build_student_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    return model

# Function to train and evaluate a model
def train_and_evaluate(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, num_epochs=12, alpha=0.5, temperature=1.0, model_name=None):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=lambda y_true, y_pred: compute_student_loss(y_true, y_pred, alpha, temperature),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=num_epochs, validation_data=(val_images, val_labels))
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    if model_name:
        model.save(model_name)  # Save the model if a model name is provided

    return test_accuracy

# Build teacher and student models
teacher_model = build_teacher_model()
student_model = build_student_model()

# Train and evaluate teacher model and save it
teacher_accuracy = train_and_evaluate(teacher_model, train_images, train_labels_one_hot, val_images, val_labels_one_hot, test_images, tf.keras.utils.to_categorical(test_labels_encoded), model_name="teacher_model.h5")

# Define a range of temperature values
temperatures = [1, 2, 4, 16, 32, 64]

# List to store student test accuracies
student_accuracies = []

for temperature in temperatures:
    # Train and evaluate student model and save it with a temperature-specific name
    student_model_name = f"student_model_T{temperature}.h5"
    student_accuracy = train_and_evaluate(student_model, train_images, train_labels_one_hot, val_images, val_labels_one_hot, test_images, tf.keras.utils.to_categorical(test_labels_encoded), temperature=temperature, model_name=student_model_name)
    student_accuracies.append(student_accuracy)

# Plot the student test accuracies vs. temperature
plt.figure(figsize=(8, 6))
plt.plot(temperatures, student_accuracies, marker='o', linestyle='-')
plt.xlabel('Temperature (T)')
plt.ylabel('Student Test Accuracy')
plt.title('Student Test Accuracy vs. Temperature')
plt.grid(True)
plt.show()

print("Teacher Model Test Accuracy:", teacher_accuracy)
print("Student Model Test Accuracies:", student_accuracies)

#%%
tf.experimental.numpy.experimental_enable_numpy_behavior()
