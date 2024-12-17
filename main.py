from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse,StreamingResponse
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import base64
from threading import Lock

app = FastAPI()

# Global variable to store the previous frame
previous_frame = None

# Allow CORS for the frontend (running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

count = 0
@app.post("/upload_frame/")
async def upload_frame(file: UploadFile = File(...), threshold_factor: int = 25, motion_factor: int = 1000 ):
    global previous_frame

    # Read the uploaded file
    contents = await file.read()
    # Convert file to numpy array
    np_arr = np.frombuffer(contents, np.uint8)
    # Decode the image
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    processed_frame = None
    thresh_image = None

    # Check if we have a previous frame to compare
    if previous_frame is not None:
        # Compute the absolute difference between current and previous frame
        frame_diff = cv2.absdiff(previous_frame, frame_gray)

        # Threshold the difference to detect motion
        _, thresh = cv2.threshold(frame_diff, threshold_factor, 255, cv2.THRESH_BINARY)

        # Dilate the threshold image to fill in holes
        dilated = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < motion_factor:
                # Ignore small contours
                continue

            # Get the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("img/"+str(count)+".jpg",frame)

        processed_frame = frame
        thresh_image = thresh

    # Update the previous frame for the next comparison
    with frame_lock:
        previous_frame = frame_gray

    if processed_frame is not None and thresh_image is not None:
        _, processed_frame_encoded = cv2.imencode('.jpg', processed_frame)
        _, thresh_image_encoded = cv2.imencode('.jpg', thresh_image)
        processed_frame_base64 = base64.b64encode(processed_frame_encoded).decode('utf-8')
        thresh_image_base64 = base64.b64encode(thresh_image_encoded).decode('utf-8')
        return {
            "message": "Motion detected",
            "processed_frame": processed_frame_base64,
            "thresh_image": thresh_image_base64
        }

    return {"message": "Frame received and processed successfully."}


frame_lock = Lock() 
# Fallback static image (replace with your static file path or a default image array)
fallback_image = cv2.imread("img/0.jpg")  # Make sure to place fallback.jpg in the same directory


def generate_stream():
    """
    A generator function to stream the latest frame.
    If the global frame is None, it streams the fallback image.
    """
    while True:
        with frame_lock:  # Ensure safe access to the previous_frame
            current_frame = previous_frame if previous_frame is not None else fallback_image

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', current_frame)
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/video")
def video_feed():
    """
    Endpoint to stream the video feed.
    If no POST request updates the frame, it will stream the fallback image.
    """
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

