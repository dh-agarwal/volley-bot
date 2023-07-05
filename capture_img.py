import cv2
import time

# Open a video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

time.sleep(.5)
# Read a frame from the webcam
ret, frame = cap.read()

if not ret:
    print("Can't receive frame")
    exit()

# Save the image
cv2.imwrite('captured_image.jpg', frame)

# Release the video capture object
cap.release()

print("Image has been successfully captured and saved.")