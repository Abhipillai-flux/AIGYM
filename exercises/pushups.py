# Import necessary libraries

import cv2
import mediapipe as mp
'''from gtts import gTTS
import os
import time
from playsound import playsound
'''
# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


'''
# Add this function to speak the counter value
def speak_counter(counter):
    tts = gTTS(text=f"You have completed {counter} push-ups.", lang='en')
    tts_file = 'counter.mp3'
    tts.save(tts_file)
    playsound(tts_file)
    os.remove(tts_file)
'''

# Open video capture
cap = cv2.VideoCapture(0)
new_width = 1280
new_height = 720

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

# Push-up counter variables
pushup_counter = 0
stage = None

# Setup MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get positions of key landmarks for push-ups
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Logic for detecting push-ups
                if left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y:
                    stage = "up"
                elif left_elbow.y > left_shoulder.y and right_elbow.y > right_shoulder.y and stage == "up":
                    stage = "down"
                    pushup_counter += 1
                    '''speak_counter(pushup_counter)'''

                    # Display counter and stage
                cv2.rectangle(image, (10, 10), (250, 150), (245, 117, 16), -1)
                cv2.putText(image, 'PUSH-UPS', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(pushup_counter),
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, stage if stage else "NA",
                            (130, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        # Render detections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Display the image
        cv2.imshow('Push-up Tracker', image)

        # Exit on 'ESC' key press or reset on 'r'
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ASCII value for ESC key
            break
        elif key == ord('r'):
            # Reset counter
            pushup_counter = 0
            stage = None

# Release resources
cap.release()
cv2.destroyAllWindows()
