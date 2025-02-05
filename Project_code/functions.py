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

def setup_environment():
    display.clear_output()
    ultralytics.checks()
    requests.packages.urllib3.disable_warnings()

def initialize_roboflow(api_key):
    return roboflow.Roboflow(api_key=api_key)

def download_dataset(rf, workspace, project, version):
    download_dir = '/content/Test-4/'
    
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Download the dataset
    dataset = rf.workspace(workspace).project(project).version(version).download("yolo11", location=download_dir)
    
    # The function will return the path to data.yaml
    return os.path.join(download_dir, 'data.yaml')

def train_model(model_path, data_config, epochs=50, imgsz=416, batch=4, device="gpu", amp=True, classes=None):
    model = YOLO(model_path)
    return model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        classes=classes
    )

def annotate_images(model, image_folder, output_folder, class_colors, class_names):
    os.makedirs(output_folder, exist_ok=True)
    for path in glob.glob(f"{image_folder}*.jpg"):
        image = cv2.resize(cv2.imread(path), (640, 640))
        results = model.predict(path)[0]
        
        for box, class_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                      results.boxes.cls.cpu().numpy(),
                                      results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            color = class_colors.get(class_id, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_names.get(class_id, 'unknown')} ({conf*100:.1f}%)",
                        (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(os.path.join(output_folder, os.path.basename(path)), image)

def evaluate_model(model):
    metrics = model.val()
    results = metrics.results_dict
    for metric, value in results.items():
        print(f"{metric}: {value}")

def display_results(output_folder):
    for path in glob.glob(f"{output_folder}*.jpg"):
        plt.imshow(plt.imread(path))
        plt.show()
