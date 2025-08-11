import cv2
import glob
from ultralytics import YOLO
import os

# Load your trained YOLO model
model = YOLO("model/best (4).pt")  # Change to your actual path

# Folder containing test images
test_images_folder = r"C:\path\to\test\images"  # Change to your path

# Loop through all images
for img_path in glob.glob(os.path.join(test_images_folder, "*.*")):
    image = cv2.imread(img_path)

    # Run YOLO inference
    results = model(image)

    # Draw bounding boxes + confidence scores
    annotated = results[0].plot()

    # Show image
    cv2.imshow("YOLO Detection", annotated)

    # Wait for key press (press 'q' to exit early)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
