# Import necessary libraries
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open video capture
cap = cv2.VideoCapture(0)
new_width = 1280
new_height = 720

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

# Jumping jacks counter variables
jumping_jack_counter = 0
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

                # Get positions of key landmarks for jumping jacks
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # Logic for detecting jumping jacks
                if left_hand.y < left_foot.y and right_hand.y < right_foot.y:
                    stage = "up"
                elif left_hand.y > left_foot.y and right_hand.y > right_foot.y and stage == "up":
                    stage = "down"
                    jumping_jack_counter += 1

                # Display counter and stage
                cv2.rectangle(image, (10, 10), (250, 150), (245, 117, 16), -1)
                cv2.putText(image, 'JUMPING JACKS', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(jumping_jack_counter),
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
        cv2.imshow('Jumping Jack Tracker', image)

        # Exit on 'ESC' key press or reset on 'r'
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ASCII value for ESC key
            break
        elif key == ord('r'):
            # Reset counter
            jumping_jack_counter = 0
            stage = None

# Release resources
cap.release()
cv2.destroyAllWindows()
