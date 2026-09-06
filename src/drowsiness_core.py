import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (128, 128)

EAR_THRESHOLD = 0.20
CNN_THRESHOLD = 0.50
CONSECUTIVE_FRAMES = 20


class DrowsinessDetector:

    def __init__(self):

        # Load trained CNN model
        self.model = tf.keras.models.load_model(
            "models/eye_state_cnn.keras"
        )

        print("CNN model loaded successfully!")

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indexes
        self.LEFT_EYE = [
            362, 385, 387, 263, 373, 380
        ]

        self.RIGHT_EYE = [
            33, 160, 158, 133, 153, 144
        ]

        # Counter for consecutive closed-eye frames
        self.closed_frames = 0


    def calculate_distance(self, point1, point2):

        return np.linalg.norm(
            point1 - point2
        )


    def calculate_ear(self, eye_points):

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = self.calculate_distance(
            p2, p6
        )

        vertical_2 = self.calculate_distance(
            p3, p5
        )

        horizontal = self.calculate_distance(
            p1, p4
        )

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        return ear


    def extract_eye(self, frame, eye_points):

        x_coordinates = [
            point[0] for point in eye_points
        ]

        y_coordinates = [
            point[1] for point in eye_points
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

        return frame[
            y_min:y_max,
            x_min:x_max
        ]


    def prepare_eye(self, eye):

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


    def predict_eye(self, eye):

        processed_eye = self.prepare_eye(
            eye
        )

        prediction = self.model.predict(
            processed_eye,
            verbose=0
        )[0][0]

        return float(prediction)


    def process_frame(self, frame):

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        status = "AWAKE"

        left_prediction = 0.0
        right_prediction = 0.0
        average_ear = 0.0

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            # -------------------------
            # LEFT EYE
            # -------------------------

            left_eye_points = []

            for index in self.LEFT_EYE:

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


            # -------------------------
            # RIGHT EYE
            # -------------------------

            right_eye_points = []

            for index in self.RIGHT_EYE:

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


            # -------------------------
            # EAR
            # -------------------------

            left_ear = self.calculate_ear(
                left_eye_points
            )

            right_ear = self.calculate_ear(
                right_eye_points
            )

            average_ear = (
                left_ear + right_ear
            ) / 2.0


            # -------------------------
            # EYE CROPS
            # -------------------------

            left_eye = self.extract_eye(
                frame,
                left_eye_points
            )

            right_eye = self.extract_eye(
                frame,
                right_eye_points
            )


            # -------------------------
            # CNN
            # -------------------------

            if left_eye.size > 0:

                left_prediction = (
                    self.predict_eye(left_eye)
                )

            if right_eye.size > 0:

                right_prediction = (
                    self.predict_eye(right_eye)
                )


            # -------------------------
            # CNN CLASSIFICATION
            # -------------------------

            left_closed = (
                left_prediction < CNN_THRESHOLD
            )

            right_closed = (
                right_prediction < CNN_THRESHOLD
            )

            cnn_closed = (
                left_closed and right_closed
            )


            # -------------------------
            # COMBINE CNN + EAR
            # -------------------------

            eyes_closed = (
                cnn_closed
                and average_ear < EAR_THRESHOLD
            )


            # -------------------------
            # FRAME COUNTER
            # -------------------------

            if eyes_closed:

                self.closed_frames += 1

            else:

                self.closed_frames = 0


            # -------------------------
            # DROWSINESS STATUS
            # -------------------------

            if (
                self.closed_frames
                >= CONSECUTIVE_FRAMES
            ):

                status = "DROWSY"

            elif eyes_closed:

                status = "EYES CLOSED"

            else:

                status = "AWAKE"


            # -------------------------
            # DRAW EYE LANDMARKS
            # -------------------------

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


            # -------------------------
            # DISPLAY INFORMATION
            # -------------------------

            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

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

            cv2.putText(
                frame,
                f"Closed Frames: {self.closed_frames}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

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

            self.closed_frames = 0

            status = "NO FACE"


            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        result = {

            "status": status,

            "ear": average_ear,

            "left_prediction":
                left_prediction,

            "right_prediction":
                right_prediction,

            "closed_frames":
                self.closed_frames
        }


        return frame, result