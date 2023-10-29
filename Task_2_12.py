#%%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
#################################################################
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# Specify the path to the 'mhist.zip' file
mhist_zip_path = 'mhist_dataset'

# Load annotations from the CSV file
annotations_path = "mhist_dataset/annotations.csv"
annotations_df = pd.read_csv(annotations_path, delimiter=',')

# Filter and split data based on the 'Partition' column
train_annotations = annotations_df[annotations_df['Partition'] == 'train']
test_annotations = annotations_df[annotations_df['Partition'] == 'test']

# Path to the directory containing the images
images_dir = "mhist_dataset/images"

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

##############################################################################

# Define the teacher model
def build_teacher_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    return model

# Define the teacher assistant model
def build_teacher_assistant_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    return model

# Define the student model
def build_student_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_encoder.classes_))
    ])
    return model

# Function to train and evaluate a model
def train_and_evaluate(model, train_images, train_labels, test_images, test_labels, num_epochs=12, model_name=None):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=num_epochs, validation_data=(test_images, test_labels))
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    if model_name:
        model.save(model_name)  # Save the model if a model name is provided

    return test_accuracy

# Build teacher, teacher assistant, and student models
teacher_model = build_teacher_model()
teacher_assistant_model = build_teacher_assistant_model()
student_model = build_student_model()

# Train and evaluate the teacher model and save it
teacher_accuracy = train_and_evaluate(teacher_model, train_images, tf.keras.utils.to_categorical(train_labels_encoded), test_images, tf.keras.utils.to_categorical(test_labels_encoded), model_name="teacher_model.h5")

# Train and evaluate the teacher assistant model
teacher_assistant_accuracy = train_and_evaluate(teacher_assistant_model, train_images, tf.keras.utils.to_categorical(train_labels_encoded), test_images, tf.keras.utils.to_categorical(test_labels_encoded))

# Train and evaluate the student model using the teacher model and teacher assistant
student_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

student_accuracy = train_and_evaluate(student_model, train_images, tf.keras.utils.to_categorical(train_labels_encoded), test_images, tf.keras.utils.to_categorical(test_labels_encoded), model_name="student_model.h5")

print("Teacher Model Test Accuracy:", teacher_accuracy)
print("Teacher Assistant Model Test Accuracy:", teacher_assistant_accuracy)
print("Student Model Test Accuracy:", student_accuracy)
