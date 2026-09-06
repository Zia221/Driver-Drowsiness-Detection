import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# MediaPipe Face Mesh
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Eye landmark indexes
# -----------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# -----------------------------
# Calculate distance
# -----------------------------

def calculate_distance(point1, point2):

    return np.linalg.norm(point1 - point2)


# -----------------------------
# Calculate EAR
# -----------------------------

def calculate_ear(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)

    horizontal = calculate_distance(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break

    height, width, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe processing
    results = face_mesh.process(rgb_frame)


    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        left_eye_points = []
        right_eye_points = []


        # -----------------------------
        # Get left eye coordinates
        # -----------------------------

        for index in LEFT_EYE:

            landmark = face_landmarks.landmark[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            left_eye_points.append(
                np.array([x, y])
            )


        # -----------------------------
        # Get right eye coordinates
        # -----------------------------

        for index in RIGHT_EYE:

            landmark = face_landmarks.landmark[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            right_eye_points.append(
                np.array([x, y])
            )


        # Calculate EAR
        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0


        # Draw eye landmarks
        for point in left_eye_points:

            cv2.circle(
                frame,
                tuple(point),
                3,
                (0, 255, 0),
                -1
            )


        for point in right_eye_points:

            cv2.circle(
                frame,
                tuple(point),
                3,
                (0, 255, 0),
                -1
            )


        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {average_ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Eye Detection + EAR",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
face_mesh.close()
cv2.destroyAllWindows()
