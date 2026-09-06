import cv2

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)

    cv2.putText(
        frame,
        f"Resolution: {width}x{height}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Driver Drowsiness - Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()