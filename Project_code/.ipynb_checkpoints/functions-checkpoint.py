''' function.py contains functions used to import our datasets from roboflow as well as train our data'''
from IPython import display
import ultralytics
import requests
import cv2
import glob
import os
from ultralytics import YOLO
import roboflow
import matplotlib.pyplot as plt
import os

dataset_path = "" 
home_dir = os.path.expanduser("~")


def setup_environment():
    display.clear_output()
    ultralytics.checks()
    requests.packages.urllib3.disable_warnings()

def initialize_roboflow(api_key):
    return roboflow.Roboflow(api_key=api_key)

def download_dataset(rf, workspace, project, version):    
    # Download the dataset
    dataset = rf.workspace(workspace).project(project).version(version).download("yolo11")
    global dataset_path 
    dataset_path = dataset.location
    # The function will return the path to data.yaml
    print('dataset downloaded')

def train_model(model_path, epochs=50, imgsz=416, batch=4, device="gpu", amp=True, classes=None):
    model = YOLO(model_path)
    model.train(
        data= dataset_path + "/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        classes=classes
    )
    print(type(model))
    return model

def annotate_images(model, class_colors, class_names):
    image_folder = os.path.join(home_dir, "Identifying-Obstruction-in-Power-Lines", "Test-4", "test", "images")
    output_folder = os.path.join(os.getcwd(), "Annotated")

    print(f"Image folder path: {image_folder}")
    print(f"Output folder path: {output_folder}")

    
    os.makedirs(output_folder, exist_ok=True)
    paths =  glob.glob(os.path.join(image_folder, "*.jpg"))
        
    for path in paths:
        # Read the image
        image = cv2.imread(path)
        image = cv2.resize(image, (640, 640))  # Resize to match model requirements
    
        results = model.predict(path)[0]
    
        # Get bounding boxes, class IDs, and confidence scores
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_names = results.names  # Dictionary of class names
        # Draw bounding boxes
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_id]
            box_color = class_colors.get(class_id, (255, 255, 255))  # Default to white if class not mapped
    
            # Draw rectangle (BGR color for OpenCV)
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    
            # Add text label (inside the box, smaller font)
            label = f"{class_name} ({conf*100:.1f}%)"
            cv2.putText(image, label,
                        (x1 + 5, y1 + 20),  # Slight padding from the top-left corner of the box
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)  # Smaller font, thinner text

        # Save the annotated image
        output_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(output_path, image)

def evaluate_model(model):
    metrics = model.val()
    results = metrics.results_dict
    for metric, value in results.items():
        print(f"{metric}: {value}")

def display_results(output_folder):
    for path in glob.glob(f"{output_folder}*.jpg"):
        plt.imshow(plt.imread(path))
        plt.show()
