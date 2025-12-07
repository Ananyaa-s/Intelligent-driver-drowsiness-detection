SnapAwake – Intelligent Driver Drowsiness Detection System

SnapAwake is an AI-powered real-time driver alertness monitoring system designed to detect drowsiness and fatigue using computer vision and deep learning. By analyzing facial landmarks, eye-closure patterns, lip distance (yawning), and adaptive thresholds generated using an LSTM model, the system helps prevent accidents caused by driver fatigue.

⭐ Key Features
🔹 Real-Time Monitoring
Continuously tracks the driver’s face using live webcam feed to detect early signs of drowsiness.

🔹 Eye Closure Detection (EAR)
Calculates the Eye Aspect Ratio (EAR) to identify prolonged eye closure, one of the strongest indicators of fatigue.

🔹 Yawn Detection
Measures lip distance to detect yawning patterns that reflect reduced alertness.

🔹 Adaptive Thresholding (LSTM Model)
Uses a trained LSTM-based deep learning model to dynamically adjust EAR and yawn thresholds based on user behavior.

🔹 Audio Alerts
Instantly plays an alarm sound through Pygame when drowsiness symptoms are detected.

🔹 Head Pose Monitoring (if enabled)
Tracks left/right head tilt to identify distraction or micro-sleep.

🔹 Mobile Notification System
Sends real-time status updates to a mobile interface via a Flask backend.

🔹 Data Logging
Saves EAR values, yawn distances, predictions, and timestamps into CSV for model retraining or analysis.
🔹 Visual Feedback

Displays real-time detection status, EAR, and yawn measurements on the screen.

🧠 Technology Stack
Technology-- Purpose
OpenC--Real-time video capture and face tracking
Dlib--68-point facial landmark detection
TensorFlow/Keras--LSTM-based adaptive threshold learning
NumPy	EAR,--lip distance, and numerical calculations
Pygame--Alarm sound playback
Flask--Backend API and minimal web interface
Socket Programming--Client–server communication for alerts
Webcam--Hardware requirement for real-time tracking

<img width="773" height="613" alt="image" src="https://github.com/user-attachments/assets/f0c77dbe-27fd-4a26-9004-09368673362e" />

▶️ How to Run the Project
1️⃣ Install dependencies
Create a virtual environment (optional but recommended):

python -m venv venv

venv\Scripts\activate   # Windows

 Install required libraries:

pip install opencv-python dlib numpy flask pygame tensorflow


(If you have a requirements.txt, use:)

pip install -r requirements.txt

2️⃣ Run the main application
python app.py

Your webcam will start, and the system will begin tracking eye movements, yawning, and thresholds.

3️⃣ Flask Web Interface
Navigate to:

http://127.0.0.1:5000/

How It Works (Brief Overview)

Face Detection → Haarcascade + Dlib
Facial Landmark Extraction → 68-point model
EAR Calculation → Detects prolonged eye closure
Lip Distance Calculation → Detects yawning
LSTM Prediction → Adaptive thresholding
System Alerts → Sound alarm & optional mobile notification
Logging → Saves data for retraining

👤 Author
Ananyaa S
