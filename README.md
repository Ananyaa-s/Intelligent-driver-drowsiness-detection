# 🚗 SnapAwake – Intelligent Driver Drowsiness Detection System

**Tech Stack:** Python, OpenCV, Dlib, TensorFlow, Flask  

SnapAwake is an AI-powered real-time driver alertness monitoring system designed to detect drowsiness and fatigue using computer vision and deep learning.

---

## ✨ Key Features

### 🔹 Real-Time Monitoring
Continuously tracks the driver’s face using a live webcam feed.

### 🔹 Eye Closure Detection (EAR)
Detects prolonged eye closure using Eye Aspect Ratio.

### 🔹 Yawn Detection
Detects yawning using lip distance.

### 🔹 Adaptive Thresholding (LSTM)
Uses an LSTM model to adjust thresholds dynamically.

### 🔹 Audio Alerts
Plays an alarm when drowsiness is detected.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
python -m venv venv
```

```powershell
venv\Scripts\activate
```

```bash
pip install opencv-python dlib numpy flask pygame tensorflow
```

---

### 2️⃣ Run the Application

```bash
python app.py
```

---

### 3️⃣ Open Web Interface

```
http://127.0.0.1:5000/
```

---

## 👤 Author
**Ananya S**
