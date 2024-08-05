import cv2
import numpy as np

# Load the pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
class_names = [
    'background', 'airplane', 'bike', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Initialize the camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for object detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{class_names[int(detections[0, 0, i, 1])]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Object Detection", frame)

    # Handle key presses
    k = cv2.waitKey(1)
    
    if k % 256 == 27:  # ESC key
        print("Escape hit")
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
