import cv2
import torch
import mediapipe as mp
from math import atan2, degrees

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')  # or yolov5m, yolov5l, yolov5x

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video source
video_path = "C:\\2024\\2nd_code\\CapstoneProject\\yoga_01.mp4"
cap = cv2.VideoCapture(video_path)

frame_skip = 5
frame_count = 0

# Angle calculation function
def calculate_angle(a, b, c):
    """
    Calculate the angle formed by points a, b, c, where b is the center.
    """
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
            (ba[0] ** 2 + ba[1] ** 2) ** 0.5 * (bc[0] ** 2 + bc[1] ** 2) ** 0.5
    )
    angle = degrees(atan2(ba[1], ba[0]) - atan2(bc[1], bc[0]))
    return abs(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Run YOLOv5 detection
    results = model(frame)

    # Extract detected bounding boxes
    detections = results.xyxy[0].numpy()  # get the bounding boxes and class
    # Filter detections for persons (class 0 in COCO dataset)
    persons = detections[detections[:, -1] == 0]  # '0' is the class ID for 'person'

    for person in persons:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, person[:4])
        cropped_person = frame[y1:y2, x1:x2]

        # Mediapipe Pose estimation on the cropped person image
        frame_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # You can now calculate angles or perform other tasks on each person's pose
            # Example: Calculate right arm angle
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_angle = calculate_angle(
                (right_shoulder.x, right_shoulder.y),
                (right_elbow.x, right_elbow.y),
                (right_wrist.x, right_wrist.y),
            )

            # Draw pose landmarks on the cropped image
            mp_drawing.draw_landmarks(cropped_person, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the angle on the image
            cv2.putText(cropped_person, f'Right Arm: {int(right_angle)} deg',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Place the cropped and processed person image back into the original frame
        frame[y1:y2, x1:x2] = cropped_person

    # Display the result
    cv2.imshow("Pose Estimation with YOLOv5", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
