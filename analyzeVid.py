import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# User inputs
arm = input("Which arm to track? Type 'left' or 'right': ").strip().lower()
lastname = input("What is the bowler's last name? ").strip().lower()
direction = input("Is the bowling action left-to-right or right-to-left? (type 'ltr' or 'rtl'): ").strip().lower()

video_file = f"{lastname}.mp4"

mp_pose = mp.solutions.pose
if arm == 'left':
    SHOULDER_LM = mp_pose.PoseLandmark.LEFT_SHOULDER
    ELBOW_LM = mp_pose.PoseLandmark.LEFT_ELBOW
    WRIST_LM = mp_pose.PoseLandmark.LEFT_WRIST
else:
    SHOULDER_LM = mp_pose.PoseLandmark.RIGHT_SHOULDER
    ELBOW_LM = mp_pose.PoseLandmark.RIGHT_ELBOW
    WRIST_LM = mp_pose.PoseLandmark.RIGHT_WRIST

cap = cv2.VideoCapture(video_file)
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

recording = False
elbow_angles = []
frames = []

prev_elbow_x = None
frame_count = 0

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Elbow Angle')
ax.axhline(15, color='red', linestyle='--', label='15° Threshold')
ax.set_ylim(0, 220)
ax.set_xlim(0, 200)
ax.set_xlabel("Frame")
ax.set_ylabel("Elbow Angle (°)")
ax.legend()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[SHOULDER_LM]
        elbow = landmarks[ELBOW_LM]
        wrist = landmarks[WRIST_LM]

        angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [elbow.x, elbow.y],
            [wrist.x, wrist.y]
        )

        if prev_elbow_x is not None:
            if direction == 'ltr':
                moving_forward = elbow.x > prev_elbow_x
            else:  # rtl
                moving_forward = elbow.x < prev_elbow_x
        else:
            moving_forward = False
        prev_elbow_x = elbow.x

        if not recording and elbow.y < shoulder.y and wrist.y < elbow.y and moving_forward:
            recording = True

        if recording:
            elbow_angles.append(angle)
            frames.append(frame_count)

            if angle <= 75:
                recording = False

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if frames:
        line.set_xdata(frames)
        line.set_ydata(elbow_angles)
        ax.set_xlim(0, max(200, frames[-1] + 10))
        plt.draw()
        plt.pause(0.001)

    cv2.imshow('Bowling Action Detection', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
# plt.show()

if len(elbow_angles) < 3:
    print("Not enough data to detect spikes.")
else:
    angles_np = np.array(elbow_angles)
    maxima_indices = argrelextrema(angles_np, np.greater)[0]
    minima_indices = argrelextrema(angles_np, np.less)[0]

    if len(maxima_indices) == 0:
        print("No maxima detected.")
    else:
        last_max = maxima_indices[-1]

        prior_minima = minima_indices[minima_indices < last_max]
        if len(prior_minima) == 0:
            last_min = 0
        else:
            last_min = prior_minima[-1]

        spike_angles = angles_np[last_min:last_max + 1]
        min_angle = np.min(spike_angles)
        max_angle = np.max(spike_angles)
        extension = max_angle - min_angle

        print(f"\nMeasured from frame {frames[last_min]} (min angle) to {frames[last_max]} (max angle)")
        print(f"Min angle: {min_angle:.2f}")
        print(f"Max angle: {max_angle:.2f}")
        print(f"Extension: {extension:.2f}")

        if extension > 15:
            print("Illegal bowling action detected.")
        else:
            print("Legal bowling action.")
