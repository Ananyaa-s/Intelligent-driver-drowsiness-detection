import cv2
import dlib
import numpy as np
import csv
from scipy.spatial import distance as dist

# Constants
EAR_THRESHOLD = 0.21
YAWN_THRESHOLD = 28.0
CSV_FILENAME = "user_data.csv"

# EAR calculation
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth distance (yawn)
def calculate_yawn_distance(mouth):
    top_lip = mouth[13]  # upper lip center
    bottom_lip = mouth[19]  # lower lip center
    return dist.euclidean(top_lip, bottom_lip)

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# Open webcam
cap = cv2.VideoCapture(0)

# CSV setup
with open(CSV_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ear", "yawn_distance", "ear_threshold", "yawn_threshold"])
    print(" Collecting data... Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            # Eyes
            left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE])
            right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE])
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            # Mouth
            mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in MOUTH])
            yawn_distance = calculate_yawn_distance(mouth)

            # Save to CSV
            writer.writerow([ear, yawn_distance, EAR_THRESHOLD, YAWN_THRESHOLD])

            # Display on screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yawn: {yawn_distance:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Collecting EAR & Yawn Data", frame)
        key = cv2.waitKey(1)
        if key != -1:
            if chr(key & 0xFF).lower() == 'q':
                break

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection complete. Check 'user_data.csv'")
