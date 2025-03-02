# %%
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

# %%
from ultralytics import YOLO

from IPython.display import display, Image

# %%
API_KEY = 'LUfenSmTeCzCw24QScjY'

# %%
import requests
requests.packages.urllib3.disable_warnings()

import roboflow
rf = roboflow.Roboflow(api_key=API_KEY)

project = rf.workspace("powerlineobstructiondetection").project("test-kpzbb")
dataset = project.version(4).download("yolo11") 

# %%
#model = YOLO('/YOLO11/yolo11n.pt')
obstruction_model = YOLO('yolo11n.pt')  # Uses current working directory


obstruction_model.train(
    data='/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/data.yaml',
    epochs=50,
    imgsz=416,
    batch=4,
    device = 'gpu',
    amp=True,
    classes=[0] #only train on obstructions
)

# %%
#model = YOLO('/YOLO11/yolo11n.pt')
pole_model = YOLO('yolo11n.pt')  # Uses current working directory


pole_model.train(
    data='/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/data.yaml',
    epochs=50,
    imgsz=416,
    batch=4,
    device = 'gpu',
    amp=True,
    classes=[1] #only train on obstructions
)

# %%
#model = YOLO('/YOLO11/yolo11n.pt')
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

# %%
def apply_nms(predictions, iou_threshold):
    kept_predictions = []
    
    while len(predictions) > 0:
        best_pred = predictions.pop(0)
        kept_predictions.append(best_pred)
        
        predictions = [pred for pred in predictions if iou(best_pred, pred) < iou_threshold or pred['class'] != best_pred['class']]
    
    return kept_predictions


# %%
def iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union


# %%
def apply_nms(results, iou_threshold=0.5):
    kept_predictions = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls = box.cls.item()
            kept_predictions.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf, 'class': cls
            })
    
    kept_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    nms_predictions = []
    
    while kept_predictions:
        best_pred = kept_predictions.pop(0)
        nms_predictions.append(best_pred)
        
        kept_predictions = [
            pred for pred in kept_predictions
            if iou(best_pred, pred) < iou_threshold or pred['class'] != best_pred['class']
        ]
    
    return nms_predictions

def iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Usage
all_predictions = pole_model.predict() + wire_model.predict() + obstruction_model.predict()
nms_predictions = apply_nms(all_predictions, iou_threshold=0.5)


# %%
import cv2
import glob
import os
import numpy as np

def annotate_images_with_nms(image_folder, output_folder, nms_predictions):
    os.makedirs(output_folder, exist_ok=True)

    class_colors = {
        'poles': (0, 255, 255),    # Yellow
        'wires': (0, 255, 0),      # Green
        'obstruction': (0, 0, 255) # Red
    }

    image_paths = glob.glob(f"{image_folder}*.jpg")

    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (640, 640))

        # Draw bounding boxes for all predictions
        for pred in nms_predictions:
            x1, y1, x2, y2 = map(int, [pred['x1'], pred['y1'], pred['x2'], pred['y2']])
            class_name = pred['class']
            conf = pred['confidence']
            box_color = class_colors.get(class_name, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            label = f"{class_name} ({conf*100:.1f}%)"
            cv2.putText(image, label,
                        (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        output_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(output_path, image)

    print(f"Annotated images saved to: {output_folder}")


# Usage
image_folder = "/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/test/images/"
output_folder = os.path.join(os.getcwd(), "Annotated_threeModelApproach")

# Assuming nms_predictions is the result from the previous NMS step
annotate_images_with_nms(image_folder, output_folder, nms_predictions)


# %%


# %%


# %%
nms_predictions[0]

# %%
import cv2
import glob
import os
import numpy as np

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

# Define the models, image folder, and output folder
models = [pole_model, wire_model, obstruction_model]
image_folder = "/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/test/images/"
output_folder = os.path.join(os.getcwd(), "Annotated_threeModelApproach")
os.makedirs(output_folder, exist_ok=True)

# Class-to-color mapping
class_colors = {
    'poles': (0, 255, 255),    # Yellow
    'wires': (0, 255, 0),      # Green
    'obstruction': (0, 0, 255) # Red
}

image_paths = glob.glob(f"{image_folder}*.jpg")

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

    # Apply NMS
    nms_predictions = apply_nms(all_predictions)

    # Draw bounding boxes
    for pred in nms_predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        class_name = pred['class']
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


# %%
import cv2
import glob
import os
import numpy as np

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

# Define the models, image folder, and output folder
models = [pole_model, wire_model, obstruction_model]
image_folder = "/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/test/images/"
output_folder = os.path.join(os.getcwd(), "Annotated_threeModelApproach_NMS_seperate")
os.makedirs(output_folder, exist_ok=True)

# Class-to-color mapping
class_colors = {
    'poles': (0, 255, 255),    # Yellow
    'wires': (0, 255, 0),      # Green
    'obstruction': (0, 0, 255) # Red
}

image_paths = glob.glob(f"{image_folder}*.jpg")

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 640))  # Resize to match model requirements

    all_predictions = []
    for model in models:
        model_predictions = []
        results = model.predict(path)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_names = results.names

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            model_predictions.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf,
                'class': class_names[class_id]
            })
        
        # Apply NMS to each model's predictions separately
        nms_model_predictions = apply_nms(model_predictions)
        all_predictions.extend(nms_model_predictions)

    # Draw bounding boxes
    for pred in all_predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        class_name = pred['class']
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


# %%
metrics = model.val()  # This will evaluate the model on the validation dataset

# Print the results
print("Validation Metrics:")

# Access results_dict to get a dictionary of all metrics
results = metrics.results_dict

# Print all metrics
for metric, value in results.items():
    print(f"{metric}: {value}")

# %%
input_tensor.shape[1] == 5

# %%
metrics = pole_model.val()  # This will evaluate the model on the validation dataset

# Print the results
print("Validation Metrics:")

# Access results_dict to get a dictionary of all metrics
results = metrics.results_dict

# Print all metrics
for metric, value in results.items():
    print(f"{metric}: {value}")

# %%
metrics = wire_model.val()  # This will evaluate the model on the validation dataset

# Print the results
print("Validation Metrics:")

# Access results_dict to get a dictionary of all metrics
results = metrics.results_dict

# Print all metrics
for metric, value in results.items():
    print(f"{metric}: {value}")

# %%
metrics = obstruction_model.val()  # This will evaluate the model on the validation dataset

# Print the results
print("Validation Metrics:")

# Access results_dict to get a dictionary of all metrics
results = metrics.results_dict

# Print all metrics
for metric, value in results.items():
    print(f"{metric}: {value}")

# %%
#'/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4'

# %%
from ultralytics import YOLO
from pathlib import Path
import torch


# Function to combine predictions and apply NMS
def combine_predictions_and_nms(models, images, iou_threshold=0.5):
    all_predictions = []
    for model in models:
        predictions = model(images)
        all_predictions.extend(predictions)
    
    # Combine predictions
    combined_pred = torch.cat([pred.boxes.data for pred in all_predictions], dim=0)
    
    # Apply NMS
    nms_predictions = torch.ops.torchvision.nms(
        combined_pred[:, :4],
        combined_pred[:, 4],
        iou_threshold
    )
    
    return combined_pred[nms_predictions]

# Validation function
def validate_combined_models(models, val_path):
    val_dataset = pole_model.datasets.YOLODataset(path=val_path, task='detect')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    combined_metrics = None
    
    for batch in val_loader:
        images, targets = batch['img'], batch['bboxes']
        nms_predictions = combine_predictions_and_nms(models, images)
        
        # Calculate metrics for this batch
        batch_metrics = pole_model.metrics.box_iou(nms_predictions[:, :4], targets[:, :4])
        
        # Accumulate metrics
        if combined_metrics is None:
            combined_metrics = batch_metrics
        else:
            for key in combined_metrics:
                combined_metrics[key] += batch_metrics[key]
    
    # Average the metrics
    for key in combined_metrics:
        combined_metrics[key] /= len(val_loader)
    
    return combined_metrics

# Use the validation function
models = [pole_model, wire_model, obstruction_model]
val_path = Path('/home/wkam/Identifying-Obstruction-in-Power-Lines/Test-4/valid')

metrics = validate_combined_models(models, val_path)

print("Validation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")



