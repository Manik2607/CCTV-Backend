import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Read the first frame to initialize the background
ret, frame1 = cap.read()
if not ret:
    print("Failed to capture video")
    exit()

# Convert to grayscale
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

while True:
    # Capture the next frame
    ret, frame2 = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Convert to grayscale
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

    # Threshold the difference to detect motion
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            # Ignore small contours
            continue

        # Get the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('img/Motion Detection.jpg', frame2)
        cv2.imwrite('img/Threshold.jpg', thresh)

    # Set the current frame as the previous frame for the next iteration
    frame1_gray = frame2_gray



# Release resources
cap.release()
cv2.destroyAllWindows()
