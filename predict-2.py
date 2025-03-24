import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("emnist_handwritten_model.keras")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# White Canvas for Drawing
canvas_width, canvas_height = 640, 480
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Drawing Settings
prev_points = []
brush_color = (0, 0, 0)  # Black brush
brush_size = 12  # Increased thickness to reduce wobbling
drawing = False
predicted_text = ""  # Store the accumulated predicted text

def preprocess_and_predict():
    global canvas, predicted_text
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary mask
    _, thresh = cv2.threshold(gray_canvas, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])

        # Add padding
        pad = 20
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = min(canvas.shape[1] - x, w + pad * 2), min(canvas.shape[0] - y, h + pad * 2)

        letter_image = thresh[y:y+h, x:x+w]

        # Resize while keeping aspect ratio
        letter_image = cv2.resize(letter_image, (28, 28), interpolation=cv2.INTER_AREA)

        # Flip & rotate to match EMNIST style
        letter_image = np.fliplr(letter_image)
        letter_image = np.rot90(letter_image)

        # Normalize
        letter_image = letter_image.astype("float32") / 255.0
        letter_image = np.expand_dims(letter_image, axis=-1)  # Add channel dimension
        letter_image = np.expand_dims(letter_image, axis=0)  # Add batch dimension

        predictions = model.predict(letter_image)
        predicted_class = np.argmax(predictions)
        predicted_letter = chr(predicted_class + 65)
        
        predicted_text += predicted_letter  # Append new letter

        # Clear the canvas after prediction
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

def is_inside_clear_button(x, y):
    return 10 <= x <= 90 and 10 <= y <= 40  # Button position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)

            # Draw cursor
            cv2.circle(frame, (index_x, index_y), 5, (0, 0, 255), -1)

            # Check if Clear button is clicked
            if is_inside_clear_button(index_x, index_y):
                canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
                predicted_text = ""

            # Distance between index and thumb
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            if distance < 30:
                drawing = True
            else:
                if drawing:
                    preprocess_and_predict()
                drawing = False
                prev_points = []

            # Scale position to canvas
            scaled_x = int(index_x * (canvas_width / w))
            scaled_y = int(index_y * (canvas_height / h))

            if drawing:
                if prev_points:
                    cv2.line(canvas, prev_points[-1], (scaled_x, scaled_y), brush_color, brush_size)
                prev_points.append((scaled_x, scaled_y))

                if len(prev_points) > 50:
                    prev_points.pop(0)

    # Draw Clear button
    cv2.rectangle(frame, (10, 10), (90, 40), (0, 0, 0), -1)
    cv2.putText(frame, "Clear", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Merge camera view and canvas
    canvas_resized = cv2.resize(canvas, (canvas_width, frame.shape[0]))
    combined_view = np.hstack((frame, canvas_resized))

    # Display status and predicted text
    status_text = "Drawing: ON" if drawing else "Drawing: OFF"
    cv2.putText(combined_view, status_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drawing else (0, 0, 255), 2)
    cv2.putText(combined_view, "Draw | Click 'Clear' | Press 'Q' to Quit", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if predicted_text:
        cv2.putText(combined_view, f"Predicted: {predicted_text}", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture-Based Notepad", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
