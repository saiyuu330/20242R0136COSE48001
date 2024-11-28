import cv2
import torch
import mediapipe as mp
from math import atan2, degrees

# Initialize YOLOv5 model
def initialize_yolov5():
    return torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')  # Use YOLOv5 small model

# Initialize Mediapipe Pose
def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_pose, pose, mp.solutions.drawing_utils

# Calculate angle between three points
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        (ba[0] ** 2 + ba[1] ** 2) ** 0.5 * (bc[0] ** 2 + bc[1] ** 2) ** 0.5
    )
    angle = degrees(atan2(ba[1], ba[0]) - atan2(bc[1], bc[0]))
    return abs(angle)

# Detect gesture based on angles and distances
def detect_gesture(landmarks, mp_pose):
    # Extract key landmarks
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
    mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

    # Calculate angles
    right_angle = calculate_angle(
        (right_shoulder.x, right_shoulder.y),
        (right_elbow.x, right_elbow.y),
        (right_wrist.x, right_wrist.y)
    )
    left_angle = calculate_angle(
        (left_shoulder.x, left_shoulder.y),
        (left_elbow.x, left_elbow.y),
        (left_wrist.x, left_wrist.y)
    )

    # Calculate normalized distances
    mouth_center = ((mouth_left.x + mouth_right.x) / 2, (mouth_left.y + mouth_right.y) / 2)
    eye_to_mouth_distance = (((left_eye.x + right_eye.x) / 2 - mouth_center[0]) ** 2 +
                             ((left_eye.y + right_eye.y) / 2 - mouth_center[1]) ** 2) ** 0.5
    right_wrist_distance = ((right_wrist.x - mouth_center[0]) ** 2 +
                            (right_wrist.y - mouth_center[1]) ** 2) ** 0.5
    left_wrist_distance = ((left_wrist.x - mouth_center[0]) ** 2 +
                           (left_wrist.y - mouth_center[1]) ** 2) ** 0.5

    normalized_right_distance = right_wrist_distance / eye_to_mouth_distance
    normalized_left_distance = left_wrist_distance / eye_to_mouth_distance

    # Detect gesture based on thresholds
    if (right_angle <= 100 and normalized_right_distance < 2.5) or \
       (left_angle <= 100 and normalized_left_distance < 2.5):
        return True
    return False

# Resize frame while maintaining aspect ratio
def resize_frame(frame, target_width=None, target_height=None):
    """
    Resize the given frame while maintaining the aspect ratio.
    Either target_width or target_height must be provided.
    """
    if target_width is None and target_height is None:
        raise ValueError("Either target_width or target_height must be specified.")

    original_height, original_width = frame.shape[:2]

    if target_width:
        scale = target_width / original_width
    else:  # If target_height is provided
        scale = target_height / original_height

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def process_video(video_path, model, mp_pose, pose, mp_drawing, target_width=None, target_height=None):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 5
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize frame to target dimensions while maintaining aspect ratio
        frame = resize_frame(frame, target_width=target_width, target_height=target_height)

        # Run YOLOv5 detection
        results = model(frame)
        detections = results.xyxy[0].numpy()
        persons = detections[detections[:, -1] == 0]  # Filter for 'person' class

        for idx, person in enumerate(persons):
            # Extract bounding box
            x1, y1, x2, y2 = map(int, person[:4])
            cropped_person = frame[y1:y2, x1:x2]

            # Pose estimation on cropped person
            frame_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame_rgb)
            is_gesture_detected = False

            if results_pose.pose_landmarks:
                is_gesture_detected = detect_gesture(results_pose.pose_landmarks.landmark, mp_pose)
                mp_drawing.draw_landmarks(cropped_person, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw bounding box and label
            color = (0, 255, 0) if is_gesture_detected else (255, 0, 0)
            label = f"Person {idx + 1}: {'Gesture' if is_gesture_detected else 'No Gesture'}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow("Pose Estimation with YOLOv5", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    video_path = "C:\\2024\\2nd_code\\CapstoneProject\\test.mp4"
    model = initialize_yolov5()
    mp_pose, pose, mp_drawing = initialize_pose()
    process_video(video_path, model, mp_pose, pose, mp_drawing, target_width=640)
