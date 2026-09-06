import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = (128, 128)

EAR_THRESHOLD = 0.20

CONSECUTIVE_FRAMES = 20

CNN_THRESHOLD = 0.50


# ==========================================
# Load trained CNN
# ==========================================

model = tf.keras.models.load_model(
    "models/eye_state_cnn.keras"
)

print("CNN model loaded successfully!")


# ==========================================
# MediaPipe Face Mesh
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# Eye landmark indexes
# ==========================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


# ==========================================
# Calculate distance
# ==========================================

def calculate_distance(point1, point2):

    return np.linalg.norm(
        point1 - point2
    )


# ==========================================
# Calculate EAR
# ==========================================

def calculate_ear(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = calculate_distance(
        p2,
        p6
    )

    vertical_2 = calculate_distance(
        p3,
        p5
    )

    horizontal = calculate_distance(
        p1,
        p4
    )

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear


# ==========================================
# Extract eye crop
# ==========================================

def extract_eye(frame, eye_points):

    x_coordinates = [
        point[0]
        for point in eye_points
    ]

    y_coordinates = [
        point[1]
        for point in eye_points
    ]

    x_min = max(
        min(x_coordinates) - 10,
        0
    )

    x_max = min(
        max(x_coordinates) + 10,
        frame.shape[1]
    )

    y_min = max(
        min(y_coordinates) - 10,
        0
    )

    y_max = min(
        max(y_coordinates) + 10,
        frame.shape[0]
    )

    eye_crop = frame[
        y_min:y_max,
        x_min:x_max
    ]

    return eye_crop


# ==========================================
# Prepare image for CNN
# ==========================================

def prepare_eye_for_cnn(eye):

    eye = cv2.resize(
        eye,
        IMAGE_SIZE
    )

    eye = cv2.cvtColor(
        eye,
        cv2.COLOR_BGR2RGB
    )

    eye = eye.astype(
        np.float32
    ) / 255.0

    eye = np.expand_dims(
        eye,
        axis=0
    )

    return eye


# ==========================================
# CNN prediction
# ==========================================

def predict_eye_state(eye):

    processed_eye = prepare_eye_for_cnn(
        eye
    )

    prediction = model.predict(
        processed_eye,
        verbose=0
    )[0][0]

    return prediction


# ==========================================
# Open camera
# ==========================================

camera = cv2.VideoCapture(0)


# ==========================================
# Drowsiness counter
# ==========================================

closed_frames = 0


# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera")

        break


    height, width, _ = frame.shape


    # ======================================
    # BGR → RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # MediaPipe
    # ======================================

    results = face_mesh.process(
        rgb_frame
    )


    status = "AWAKE"


    if results.multi_face_landmarks:

        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        # ==================================
        # Extract left eye points
        # ==================================

        left_eye_points = []

        for index in LEFT_EYE:

            landmark = (
                face_landmarks.landmark[index]
            )

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            left_eye_points.append(
                np.array([x, y])
            )


        # ==================================
        # Extract right eye points
        # ==================================

        right_eye_points = []

        for index in RIGHT_EYE:

            landmark = (
                face_landmarks.landmark[index]
            )

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            right_eye_points.append(
                np.array([x, y])
            )


        # ==================================
        # Calculate EAR
        # ==================================

        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0


        # ==================================
        # Extract eye crops
        # ==================================

        left_eye = extract_eye(
            frame,
            left_eye_points
        )

        right_eye = extract_eye(
            frame,
            right_eye_points
        )


        # ==================================
        # CNN predictions
        # ==================================

        left_prediction = (
            predict_eye_state(left_eye)
        )

        right_prediction = (
            predict_eye_state(right_eye)
        )


        # ==================================
        # Determine CNN eye state
        # ==================================

        left_closed = (
            left_prediction < CNN_THRESHOLD
        )

        right_closed = (
            right_prediction < CNN_THRESHOLD
        )


        cnn_closed = (
            left_closed and right_closed
        )


        # ==================================
        # Combine CNN + EAR
        # ==================================

        eyes_closed = (
            cnn_closed
            and average_ear < EAR_THRESHOLD
        )


        # ==================================
        # Temporal logic
        # ==================================

        if eyes_closed:

            closed_frames += 1

        else:

            closed_frames = 0


        # ==================================
        # Drowsiness decision
        # ==================================

        if closed_frames >= CONSECUTIVE_FRAMES:

            status = "DROWSY"

        elif eyes_closed:

            status = "EYES CLOSED"

        else:

            status = "AWAKE"


        # ==================================
        # Draw landmarks
        # ==================================

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


        # ==================================
        # Display EAR
        # ==================================

        cv2.putText(
            frame,
            f"EAR: {average_ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==================================
        # Display CNN predictions
        # ==================================

        cv2.putText(
            frame,
            f"Left CNN: {left_prediction:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Right CNN: {right_prediction:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==================================
        # Display frame counter
        # ==================================

        cv2.putText(
            frame,
            f"Closed Frames: {closed_frames}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==================================
        # Display status
        # ==================================

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ======================================
    # Display frame
    # ======================================

    cv2.imshow(
        "CNN Driver Drowsiness Detection",
        frame
    )


    # ======================================
    # Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# Cleanup
# ==========================================

camera.release()

face_mesh.close()

cv2.destroyAllWindows()