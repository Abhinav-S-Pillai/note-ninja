# 📝 Note Ninja: Gesture-Based Handwriting Recognition Notepad

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Drawbacks & Limitations](#drawbacks--limitations)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Overview

**Note Ninja** is an AI-powered, gesture-based notepad that lets you write letters in the air using your hand and a webcam. It uses real-time hand tracking (MediaPipe) and a deep learning model (trained on EMNIST) to recognize handwritten characters, converting your gestures into digital text.

This project is designed to showcase practical skills in computer vision, deep learning, and intuitive user interfaces. Whether you're a beginner or an experienced developer, this project is easy to set up and run.

---

## Features

- **Real-Time Hand Tracking:** Uses your webcam and MediaPipe to detect hand landmarks.
- **Gesture-Based Drawing:** Draw on a virtual canvas by pinching your index finger and thumb.
- **Handwritten Character Recognition:** Recognizes drawn letters using a trained neural network (EMNIST).
- **Live Feedback:** See your drawing, hand landmarks, and predictions instantly.
- **User-Friendly Controls:** Clear the canvas with a gesture, and quit with a key press.
- **Beginner-Friendly Setup:** Minimal dependencies and simple instructions to get started.

---

## How It Works

1. **Hand Detection:**  
   MediaPipe detects your hand and tracks 21 landmarks in real time.

2. **Drawing Gesture:**  
   When you pinch your index finger and thumb together, the app starts drawing on a virtual canvas at the tip of your index finger.

3. **Character Recognition:**  
   When you stop pinching, the drawn image is preprocessed and sent to a neural network trained on the EMNIST dataset, which predicts the character.

4. **User Interface:**  
   The app displays the webcam feed, hand landmarks, drawing canvas, and predicted text side by side for a seamless experience.

---

## Demo

> **Tip:** Add your own GIF or screenshot here to visually demonstrate the app in action.

---

## Installation

### 1. **Clone the Repository**

```sh
git clone https://github.com/yourusername/note-ninja.git
cd note-ninja
```

### 2. **Install Dependencies**

Make sure you have Python 3.8+ installed. Run the following command to install all required libraries:

```sh
pip install -r requirements.txt
```

### 3. **Download the Model**

Place your trained model file `emnist_handwritten_model.keras` in the project root directory.

> **Note:** If you don't have the model, you can train one using the EMNIST dataset. (Instructions can be provided if needed.)

---

## Usage

1. **Connect your webcam.**
2. **Run the application:**

   ```sh
   python predict-2.py
   ```

3. **How to Use:**
   - **Draw:** Pinch your index finger and thumb together and move your hand to draw on the virtual canvas.
   - **Clear:** Move your index finger to the "Clear" button on the screen.
   - **Predict:** When you stop pinching, the app will automatically recognize the drawn character and display it.
   - **Quit:** Press `Q` on your keyboard to exit the application.

---

## Project Structure

```
note-ninja/
│
├── emnist_handwritten_model.keras   # Trained Keras model for character recognition
├── predict-2.py                    # Main application script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Troubleshooting

- **Webcam Not Detected:**  
  Ensure your webcam is connected and not being used by another application.

- **No Prediction Output:**  
  Make sure your drawing is clear and covers enough area. The model may not recognize ambiguous shapes.

- **Model File Missing:**  
  Download or train `emnist_handwritten_model.keras` and place it in the project root.

- **Dependency Issues:**  
  Run `pip install -r requirements.txt` to ensure all dependencies are installed.

- **MediaPipe or TensorFlow Errors:**  
  Some versions of MediaPipe and TensorFlow may have compatibility issues. If you encounter errors, try upgrading or downgrading these packages.

- **Application Crashes or Freezes:**  
  This can happen if your system is low on resources or if the webcam feed is interrupted. Restart the application and ensure no other apps are using the webcam.

---

## Drawbacks & Limitations

- **Lighting Conditions:**  
  The accuracy of hand tracking and gesture recognition can be affected by poor lighting or cluttered backgrounds.

- **Single-Handed Operation:**  
  The app is designed for single-hand use and may not work well if multiple hands are visible.

- **Model Limitations:**  
  The recognition model is trained on EMNIST and may not recognize all handwriting styles, especially if the drawn letter is unclear or not centered.

- **Hardware Requirements:**  
  Requires a functional webcam and a moderately powerful CPU for real-time processing.

- **Platform Compatibility:**  
  Tested primarily on Windows. Some dependencies may require additional setup on Mac or Linux.

- **Potential Errors:**  
  - If the webcam is not detected, the app will not start.
  - If the model file is missing or corrupted, predictions will not work.
  - In rare cases, MediaPipe or TensorFlow may throw errors due to version mismatches or missing system libraries.
  - If the hand is not detected, drawing and prediction will not function.

- **False Positives:**  
  Sometimes, accidental pinches or hand movements may trigger drawing or clearing actions unintentionally.

---

## Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## Contact

**Author:** Abhinav S pillai  
**Email:** abhinavspillai2005@gmail.com  
**GitHub:** Abhinav-S-Pillai

---

> **Note for Beginners:**  
> This project is designed to be beginner-friendly. Follow the installation and usage instructions carefully, and you'll be able to run the app without any issues. If you encounter problems, refer to the troubleshooting section or contact the author.


> This project demonstrates practical skills in computer vision, deep learning, real-time user interfaces, and Python development. The code is well-commented and modular, making it easy to extend or adapt for other gesture-based applications.
