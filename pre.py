import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the EMNIST dataset from the .mat file
emnist = scipy.io.loadmat('P:\programming\projects\sign ninja\matlab\emnist-letters.mat')

# Extract training and testing data
train_images = emnist['dataset'][0][0][0][0][0][0]  # Training images
train_labels = emnist['dataset'][0][0][0][0][0][1].flatten()  # Training labels
test_images = emnist['dataset'][0][0][1][0][0][0]  # Testing images
test_labels = emnist['dataset'][0][0][1][0][0][1].flatten()  # Testing labels

# Reshape images to 28x28 (EMNIST is stored in a single-row format)
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert labels to categorical (One-Hot Encoding)
num_classes = len(np.unique(train_labels))
train_labels = tf.keras.utils.to_categorical(train_labels - 1, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels - 1, num_classes)

# Split train into train and validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

print("✅ EMNIST Preprocessing Complete!")
print(f"📂 Training Samples: {train_images.shape[0]}")
print(f"📂 Validation Samples: {val_images.shape[0]}")
print(f"📂 Testing Samples: {test_images.shape[0]}")
