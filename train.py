import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pre import train_images, train_labels, val_images, val_labels, test_images, test_labels  # Import preprocessed data

print("✅ Starting training for Handwritten Letters...")

# Define a CNN model for EMNIST
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Prevents overfitting
    Dense(26, activation='softmax')  # 26 classes for A-Z letters
])

print("✅ Model architecture created!")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("✅ Model compiled! Training starts now...")

# Callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=20, 
                    batch_size=64,
                    callbacks=[early_stop, reduce_lr])

# Save trained model
model.save("emnist_handwritten_model.keras")
model.save("emnist_handwritten_model.h5")

print("✅ Model training complete! Model saved.")

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")
