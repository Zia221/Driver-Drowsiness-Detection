import cv2
import mediapipe as mp

# Create MediaPipe Face Mesh object
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open laptop camera
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = face_mesh.process(rgb_frame)

    # Check whether a face was detected
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            height, width, _ = frame.shape

            # Draw all facial landmarks
            for landmark in face_landmarks.landmark:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow("Face & Eye Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()