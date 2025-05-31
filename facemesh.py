import cv2
import mediapipe as mp

print("Import successful!")

# Initialize mediapipe drawing and face mesh utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Function to draw face mesh
def drawFaceMesh(image, results):
    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)    # width
cap.set(4, 420)    # height
cap.set(10, 100)   # brightness

# Create FaceMesh instance once
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        success, frame = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = face_mesh.process(rgb_frame)

        # Draw the face mesh on original BGR frame
        drawFaceMesh(frame, results)

        cv2.imshow("FaceMesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
