import cv2
import pyautogui
import numpy as np

# Set resolution and output file details
resolution = pyautogui.size()
codec = cv2.VideoWriter_fourcc(*"XVID")
filename = "screen_recording.avi"
fps = 15.0

# Create VideoWriter object
out = cv2.VideoWriter(filename, codec, fps, resolution)

# Optional: Display recording in real-time
cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recording", 480, 270)

print("Recording... Press 'q' to stop.")

while True:
    # Capture screenshot
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write frame to video file
    out.write(frame)

    # Show live preview
    cv2.imshow("Recording", frame)

    # Stop recording on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()
print("Recording saved as", filename)