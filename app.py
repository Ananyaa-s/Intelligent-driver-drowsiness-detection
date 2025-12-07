# app.py (Final Integrated Version)
from flask import Flask, render_template, jsonify
import threading
import cv2
import csv
import numpy as np
import dlib
from imutils import face_utils
import pygame
from collections import deque
import traceback
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)
status_data = {'status': 'Initializing'}

pygame.mixer.init()

# Global Variables
sleep = 0
yawn_counter = 0
alarm_counter = 0
EAR_list = []
YAWN_list = []
TILT_list = []
yawns = 0
status = ""
eye_closed_frames = 0
yawn_frames = 0
blink_sets = 0
YAWN_OFFSET = 8
yawn_sets = 0
SEQUENCE_LENGTH = 10
sequence = []
drowsiness_alert_triggered = False
# Load adaptive model (pre-trained LSTM or similar)
model = load_model("snapawake_lstm_3features.h5")  # use new model
model.summary()
recent_data = []  # store latest blink/yawn/head data for prediction

# Thresholds (will be adapted dynamically)
EAR_THRESHOLD = 0.28
# Messages
message1 = "Driver is sleepy"
message2 = "Driver is very drowsy"

def play_audio():
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load('alert.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    except pygame.error as e:
        print(f"Error playing audio: {e}")

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    return abs(np.mean(top_lip[:, 1]) - np.mean(low_lip[:, 1]))
       
def eye_aspect_ratio(eye):
    A = compute(eye[1], eye[5])
    B = compute(eye[2], eye[4])
    C = compute(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def alarms():
    global alarm_counter, yawns, status_data
    if alarm_counter >= 2 and yawns == 2:
        status_data['status'] = message1
        print(message1)
        play_audio()
    elif alarm_counter > 5 and yawns >= 3:
        status_data['status'] = message2
        print(message2)
        play_audio()
        alarm_counter = 0

def adapt_threshold():
    try:
        df = pd.read_csv("user_data.csv")

        if df.empty or df.shape[0] < 30:
            print("ℹNot enough valid data to run threshold adaptation.")
            return

        features = df[['ear', 'yawn_distance']].values
        input_seq = features[-30:].reshape(1, 30, 2)

        prediction = model.predict(input_seq, verbose=0)

        # Full debug print
        print(" Prediction raw output:", prediction)
        print(" Shape:", prediction.shape, "| Type:", type(prediction))

        # Try to extract the value
        try:
            value = float(prediction.squeeze())
            global EAR_THRESHOLD
            EAR_THRESHOLD = value
            # Constrain threshold to a realistic minimum
            EAR_THRESHOLD = max(0.25, min(EAR_THRESHOLD, 0.30))  # keep it within reasonable bounds
            print(f"Adjusted EAR_THRESHOLD to {EAR_THRESHOLD:.3f}")

            print(f"Updated EAR_THRESHOLD to {EAR_THRESHOLD:.3f}")
        except Exception as e:
            print(f" Prediction unpacking failed: {e} | Output: {repr(prediction)}")

    except Exception as e:
        print(f"Error in adapt_threshold(): {e}")


def detect_drowsiness():
    global sleep, yawn_counter, alarm_counter, yawns, status_data, status, recent_data
    global EAR_list, YAWN_list, TILT_list
    global blink_sets, yawn_sets, drowsiness_alert_triggered
    global EAR_THRESHOLD, YAWN_THRESHOLD
    global eye_closed_frames, yawn_frames
    global head_tilt_left_count, head_tilt_right_count
    head_tilt_left_count = 0
    head_tilt_right_count = 0
    last_tilt_direction = None 
    prediction = None
    probability = None
    lstm_output = None


    sequence_buffer = deque(maxlen=30)  
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" Error: Could not open webcam.")
            status_data['status'] = 'Camera Error'
            return

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # Draw green circles for each eye landmark
       
        # Calibration phase
        print("🔍 Calibrating normal lip distance...")
        calibration_distances = []
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                continue  # skip if frame not captured

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if faces:
                landmarks = predictor(gray, faces[0])
                landmarks = face_utils.shape_to_np(landmarks)
                calibration_distances.append(lip_distance(landmarks))

        if len(calibration_distances) == 0:
            print(" Error: No face detected during calibration.")
            cap.release()
            cv2.destroyAllWindows()
            return

        normal_lip_distance = np.mean(calibration_distances)
        YAWN_THRESHOLD = normal_lip_distance + YAWN_OFFSET  # e.g., YAWN_OFFSET = 8
        print(f"Calibration complete. Normal lip distance: {normal_lip_distance:.2f}")
        EAR_THRESHOLD = 0.28
        YAWN_THRESHOLD = normal_lip_distance + YAWN_OFFSET
        print(f" Initial EAR_THRESHOLD = {EAR_THRESHOLD:.3f}")
        print(f"Initial YAWN_THRESHOLD = {YAWN_THRESHOLD:.3f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Skipped a frame due to capture error.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)
                for i in range(36, 48):  # 36–41: left eye, 42–47: right eye
                    x, y = landmarks[i]
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Head tilt (left/right) detection with count
                # Head Tilt Angle Debug
                nose = landmarks[33]
                left_cheek = landmarks[1]
                right_cheek = landmarks[15]

                dx = right_cheek[0] - left_cheek[0]
                dy = right_cheek[1] - left_cheek[1]
                angle = np.degrees(np.arctan2(dy, dx))

                print(f"Head Tilt Angle: {angle:.2f}, Last: {last_tilt_direction}")

                if angle > 10 and last_tilt_direction != "right":
                    head_tilt_right_count += 1
                    last_tilt_direction = "right"
                    print(f" Head Tilt Right Count: {head_tilt_right_count}")
                elif angle < -10 and last_tilt_direction != "left":
                    head_tilt_left_count += 1
                    last_tilt_direction = "left"
                    print(f" Head Tilt Left Count: {head_tilt_left_count}")
                elif -10 <= angle <= 10:
                    last_tilt_direction = None  # Reset when head is straight

                # Display on frame
                cv2.putText(frame, f"Head Tilt Left: {head_tilt_left_count}", (50, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Head Tilt Right: {head_tilt_right_count}", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Compute actual EAR values
                left_ear = eye_aspect_ratio([landmarks[i] for i in range(36, 42)])
                right_ear = eye_aspect_ratio([landmarks[i] for i in range(42, 48)])
                ear = (left_ear + right_ear) / 2.0
                cv2.putText(frame, f"EAR: {ear:.3f}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"EAR: {ear:.3f} | Threshold: {EAR_THRESHOLD}")

# Now detect blink based on threshold
                left_blink = left_ear <= EAR_THRESHOLD
                right_blink = right_ear <= EAR_THRESHOLD

                # Yawn distance
                distance = lip_distance(landmarks)

                tilt_count = 0  # Placeholder for head tilt
                # Append to sequence buffer
                print(f" EAR: {ear:.3f} | Threshold: {EAR_THRESHOLD:.3f} | EAR Frames: {sleep}")


                sequence_buffer.append([ear, distance])
                if len(sequence_buffer) == 30:
                    input_seq = np.array(sequence_buffer).reshape(1, 30, 2)
                    prediction = model.predict(input_seq, verbose=0)
                    probability = prediction[0][0]  # since shape is (1, 1)
                    lstm_output = probability
                    print(f" LSTM Probability: {probability:.4f}")
                    if probability > 0.3:
                        print("  LSTM predicted: Drowsy")
                        if not drowsiness_alert_triggered:
                            drowsiness_alert_triggered = True
                            status_data['status'] = 'Drowsy'
                            threading.Thread(target=play_audio).start()
                            cv2.putText(frame, status_data['status'], (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        print(" LSTM predicted: Awake")
                        if not drowsiness_alert_triggered:
                            status_data['status'] = ' Awake'



                #  Append EAR and Yawn Distance to CSV live
                with open("user_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ear, distance, EAR_THRESHOLD, YAWN_THRESHOLD])

                EAR_list.append(ear)
                YAWN_list.append(distance)
                #TILT_list.append(tilt_count)

                if len(EAR_list) > 100:
                    EAR_list.pop(0)
                if len(YAWN_list) > 100:
                    YAWN_list.pop(0)
                # Save for ML-based adaptation
                recent_data.append([sleep, yawns, distance])
                if len(recent_data) > 100:
                    recent_data = recent_data[-100:]

                if len(recent_data) >= 30:
                    adapt_threshold()

                # Detection logic
                if left_blink or right_blink:
                    eye_closed_frames += 1
                    if eye_closed_frames >= 20:
                        blink_sets += 1
                        print(f" Blink set #{blink_sets} (20+ frames)")
                        eye_closed_frames = 0
                else:
                    eye_closed_frames = 0 

                if distance > YAWN_THRESHOLD:

                    yawn_frames += 1
                    if yawn_frames >= 10:
                        yawn_sets += 1
                        print(f" Yawn set #{yawn_sets} (10+ frames)")
                        yawn_frames = 0
                else:
                    yawn_frames = 0
            #  Combined Drowsiness Detection
           #  Final Drowsiness Decision (Rule + LSTM)
            if lstm_output is not None:
                print(f" LSTM Output: {lstm_output:.3f} | Blink Sets: {blink_sets} | Yawn Sets: {yawn_sets}")
            else:
                print(f" LSTM Output: None | Blink Sets: {blink_sets} | Yawn Sets: {yawn_sets}")

            print(f" LSTM Output: {lstm_output if lstm_output is not None else 'None'} | Blink Sets: {blink_sets} | Yawn Sets: {yawn_sets}")

            if (lstm_output is not None and lstm_output > 0.5 or (blink_sets >= 2 and yawn_sets >= 2)) and not drowsiness_alert_triggered:
                drowsiness_alert_triggered = True
                status_data['status'] = 'Drowsy!!!'
                print(" ALERT: Drowsiness Detected!")
                threading.Thread(target=play_audio).start()
                cv2.putText(frame, status_data['status'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"EAR count: {sleep}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawns: {yawns}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if faces:
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    # Reset logic
            if drowsiness_alert_triggered and blink_sets < 2 and yawn_sets < 2:
                 drowsiness_alert_triggered = False
                 status_data['status'] = 'Awake'
                 print(" Driver back to normal.")
            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" Quitting...")
                break
    except Exception as e:
        print(f" Exception occurred: {e}")
        traceback.print_exc()

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Save captured features for ML
        min_len = min(len(EAR_list), len(YAWN_list))
        df = pd.DataFrame({
            'ear': EAR_list[:min_len],
            'yawn_distance': YAWN_list[:min_len]
        })
        # Only add thresholds if YAWN_THRESHOLD was set
        if 'YAWN_THRESHOLD' in globals():
             df['ear_threshold'] = EAR_THRESHOLD
             df['yawn_threshold'] = YAWN_THRESHOLD
        else:
              print(" Skipping threshold columns — YAWN_THRESHOLD not defined.")
        df.to_csv('user_data.csv', index=False)
        print(" Data saved to user_data.csv")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(status_data)

if __name__ == '__main__':
    threading.Thread(target=detect_drowsiness, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
