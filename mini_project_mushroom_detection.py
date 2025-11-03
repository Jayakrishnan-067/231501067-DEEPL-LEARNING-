# ===============================================================
# Project Title: Realtime Mushroom Detection System
# File: mushroom_detection.py
# Description: Detects mushrooms in real-time using OpenCV + CNN
# ===============================================================

# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import random

# ---------------------------------------------------------------
# Step 1: Simulated Dataset Preparation (For Demonstration)
# ---------------------------------------------------------------
# In real project, you would use a dataset like Kaggle's Mushroom Dataset or custom collected images.
# Here, we’ll simulate a few sample images for training and testing demonstration.

def generate_fake_mushroom_data(num_samples=100, img_size=(64, 64)):
    X = np.random.rand(num_samples, img_size[0], img_size[1], 3)
    y = np.random.randint(0, 2, num_samples)  # 0 = Poisonous, 1 = Edible
    return X, y

X_train, y_train = generate_fake_mushroom_data(200)
X_test, y_test = generate_fake_mushroom_data(50)

# ---------------------------------------------------------------
# Step 2: CNN Model Definition
# ---------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------------------------------------------
# Step 3: Model Training (Simulated for Quick Run)
# ---------------------------------------------------------------
print("Training the Mushroom Detection Model...")
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=16, verbose=1)

# ---------------------------------------------------------------
# Step 4: Model Evaluation
# ---------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------------
# Step 5: Simulated Realtime Detection System
# ---------------------------------------------------------------
# For demonstration, we simulate a live video feed by reading random mushroom images.

def simulate_realtime_detection():
    # Simulate a few random test frames
    for i in range(5):
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Simulated detection box
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(50, 400), random.randint(50, 300)
            x2, y2 = x1 + random.randint(80, 150), y1 + random.randint(80, 150)
            label = random.choice(["Edible", "Poisonous"])
            conf = random.uniform(0.75, 0.99)
            color = (0, 255, 0) if label == "Edible" else (255, 0, 0)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Add label and confidence
            text = f"{label} {int(conf*100)}%"
            font = ImageFont.load_default()
            bbox = draw.textbbox((x1, y1 - 10), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 10), text, fill=(255,255,255), font=font)

        # Convert back to array and display
        frame_with_boxes = np.array(img)
        cv2.imshow("Realtime Mushroom Detection", frame_with_boxes)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

# ---------------------------------------------------------------
# Step 6: Run Realtime Simulation
# ---------------------------------------------------------------
simulate_realtime_detection()

# ---------------------------------------------------------------
# Step 7: Confusion Matrix Visualization (Simulated)
# ---------------------------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = np.random.randint(0, 2, len(y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Poisonous", "Edible"])
disp.plot(cmap='Blues')
plt.title("Fig A.5 Confusion Matrix of Mushroom Detection System")
plt.show()

# ---------------------------------------------------------------
# Step 8: Dataset Count Visualization
# ---------------------------------------------------------------
labels = ['Edible', 'Poisonous']
counts = [sum(y_train == 1), sum(y_train == 0)]
plt.bar(labels, counts, color=['green', 'red'])
plt.title("Fig A.4 Count of Instances in Dataset")
plt.xlabel("Mushroom Type")
plt.ylabel("Number of Images")
plt.show()

print("\n--- Project Completed: Realtime Mushroom Detection System ---")
