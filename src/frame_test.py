import cv2

camera = cv2.VideoCapture(0)

success, frame = camera.read()

print("Success:", success)
print("Frame type:", type(frame))
print("Frame shape:", frame.shape)

camera.release()