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
    return model

def train_nms_model(model_path, epochs=50, imgsz=416, batch=4, device="gpu", amp=True, classes=None):
    obstruction_model = YOLO(model_path)
    obstruction_model.train(
        data= dataset_path + "/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        classes=[0]
    )
    
    pole_model = YOLO('yolo11n.pt')  
    pole_model.train(
        data='/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/data.yaml',
        epochs=50,
        imgsz=416,
        batch=4,
        device = 'gpu',
        amp=True,
        classes=[1] 
    )
    wire_model = YOLO('yolo11n.pt')  # Uses current working directory
    
    
    wire_model.train(
        data='/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/data.yaml',
        epochs=50,
        imgsz=416,
        batch=4,
        device = 'gpu',
        amp=True,
        classes=[2] #only train on obstructions
    )
    models = [pole_model, wire_model, obstruction_model]
    return models

def apply_nms(predictions, iou_threshold=0.3):
    kept_predictions = []
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    while predictions:
        best_pred = predictions.pop(0)
        kept_predictions.append(best_pred)
        
        predictions = [pred for pred in predictions if iou(best_pred, pred) < iou_threshold or pred['class'] != best_pred['class']]
    
    return kept_predictions

def iou(box1, box2):
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
    
def annotate_images_NMS(models, class_colors):
    image_folder = os.path.join(home_dir, "Identifying-Obstruction-in-Power-Lines", "Test-4", "test", "images")
    output_folder = os.path.join(os.getcwd(), "Annotated_NMSApproach")

    print(f"Image folder path: {image_folder}")
    print(f"Output folder path: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
   
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (640, 640))  # Resize to match model requirements

        all_predictions = []
        for model in models:
            results = model.predict(path)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_names = results.names

            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                all_predictions.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'class': class_names[class_id]
                })

        # Apply NMS (you need to implement this function)
        nms_predictions = apply_nms(all_predictions)

        # Draw bounding boxes
        for pred in nms_predictions:
            x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
            class_name = pred['class']
            print(class_name)
            print(class_colors)
            conf = pred['confidence']
            box_color = class_colors.get(class_name, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            label = f"{class_name} ({conf*100:.1f}%)"
            cv2.putText(image, label,
                        (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Save the annotated image
        output_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(output_path, image)

    print(f"Annotated images saved to: {output_folder}")

def evaluate_model_nms(models):
    for model in models:
        metrics = model.val()  # Evaluate the model on the validation dataset
        results = metrics.results_dict

        # Print all metrics
        for metric, value in results.items():
            print(f"{metric}: {value}")
        
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
