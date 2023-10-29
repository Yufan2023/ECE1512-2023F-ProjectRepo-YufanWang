import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the teacher model
def build_teacher_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define the student model
def build_student_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
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

# Build teacher and student models
teacher_model = build_teacher_model()
student_model = build_student_model()

# Train and evaluate teacher model and save it
teacher_accuracy = train_and_evaluate(teacher_model, train_images, tf.keras.utils.to_categorical(train_labels), test_images, tf.keras.utils.to_categorical(test_labels), model_name="teacher_model.h5")

# Train and evaluate student model using the teacher model and teacher assistant
student_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

student_accuracy = train_and_evaluate(student_model, train_images, tf.keras.utils.to_categorical(train_labels), test_images, tf.keras.utils.to_categorical(test_labels), model_name="student_model.h5")

print("Teacher Model Test Accuracy:", teacher_accuracy)
print("Student Model Test Accuracy:", student_accuracy)
