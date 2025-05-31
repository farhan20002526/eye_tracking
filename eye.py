import cv2
import mediapipe as mp
import math

print("Import successful!")

# Initialize mediapipe drawing and face mesh utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Iris landmark indices
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

def euclidean_distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def get_eye_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx, image_shape):
    h, w = image_shape[:2]
    top = int(landmarks[top_idx].x * w), int(landmarks[top_idx].y * h)
    bottom = int(landmarks[bottom_idx].x * w), int(landmarks[bottom_idx].y * h)
    left = int(landmarks[left_idx].x * w), int(landmarks[left_idx].y * h)
    right = int(landmarks[right_idx].x * w), int(landmarks[right_idx].y * h)

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)
    ear = vertical / horizontal if horizontal != 0 else 0
    return ear

def get_iris_center(landmarks, iris_indices, image_shape):
    h, w = image_shape[:2]
    x = int(sum([landmarks[i].x for i in iris_indices]) / len(iris_indices) * w)
    y = int(sum([landmarks[i].y for i in iris_indices]) / len(iris_indices) * h)
    return x, y

def get_eye_box_and_gaze(landmarks, eye_indices, iris_indices, image_shape):
    h, w = image_shape[:2]
    eye_x1 = int(landmarks[eye_indices[0]].x * w)
    eye_x2 = int(landmarks[eye_indices[1]].x * w)
    eye_y1 = int(landmarks[eye_indices[0]].y * h)
    eye_y2 = int(landmarks[eye_indices[1]].y * h)
    iris_x, iris_y = get_iris_center(landmarks, iris_indices, image_shape)
    cv2.rectangle(frame, (eye_x1, eye_y1 - 10), (eye_x2, eye_y2 + 10), (0, 255, 255), 1)

    eye_width = eye_x2 - eye_x1
    if eye_width == 0:
        return "Unknown"

    iris_position_ratio = (iris_x - eye_x1) / eye_width
    if iris_position_ratio < 0.35:
        return "Looking Left"
    elif iris_position_ratio > 0.65:
        return "Looking Right"
    else:
        return "Looking Center"

# Blink tracking variables
blink_count = 0
blinked = False

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)
cap.set(10, 100)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark

            left_iris_x, left_iris_y = get_iris_center(mesh_points, LEFT_IRIS, frame.shape)
            right_iris_x, right_iris_y = get_iris_center(mesh_points, RIGHT_IRIS, frame.shape)
            cv2.circle(frame, (left_iris_x, left_iris_y), 2, (255, 0, 255), -1)
            cv2.circle(frame, (right_iris_x, right_iris_y), 2, (255, 0, 255), -1)

            left_gaze = get_eye_box_and_gaze(mesh_points, [33, 133], LEFT_IRIS, frame.shape)
            right_gaze = get_eye_box_and_gaze(mesh_points, [362, 263], RIGHT_IRIS, frame.shape)

            left_ear = get_eye_aspect_ratio(mesh_points, 159, 145, 33, 133, frame.shape)
            right_ear = get_eye_aspect_ratio(mesh_points, 386, 374, 362, 263, frame.shape)

            left_eye_state = "Closed" if left_ear < 0.20 else "Open"
            right_eye_state = "Closed" if right_ear < 0.20 else "Open"

            cv2.putText(frame, f"Left Eye: {left_eye_state}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Right Eye: {right_eye_state}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Left Eye: {left_gaze}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {right_gaze}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Blink detection logic
            if left_eye_state == "Closed" and right_eye_state == "Closed":
                if not blinked:
                    blink_count += 1
                    blinked = True
            else:
                blinked = False

            # Show blink count
            cv2.putText(frame, f"Blinks: {blink_count}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Full Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
