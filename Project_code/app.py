import gradio as gr
import torch
import cv2  # OpenCV for video processing
import numpy as np  # For handling image arrays
from PIL import Image, ImageDraw
from ultralytics import YOLO
import uuid
import torchvision.transforms as transforms  # Import transforms
import os

# Load your trained model
model_path = "Project_code/runs/detect/train13/weights/best.pt"  # Adjust this path if necessary
model = YOLO(model_path)  # Load the YOLO model directly
model.eval()  # Set the model to evaluation mode

def process_image(input_image):
    # Convert from numpy array to PIL Image
    image = Image.fromarray(input_image)
    
    # Prepare image for model
    image_tensor = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])(image).unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process outputs and draw boxes
    boxes = outputs[0].boxes.xyxy
    scores = outputs[0].boxes.conf
    classes = outputs[0].boxes.cls

    # Draw predictions on image
    if len(boxes) > 0:
        draw = ImageDraw.Draw(image)
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # Add label with class and confidence
            label = f"Class {int(cls)} ({score:.2f})"
            draw.text((x1, y1-10), label, fill="red")

    # Convert back to numpy array
    return np.array(image)

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=gr.Image(label="Detected Objects"),
    title="Power Line Obstruction Detection",
    description="Upload an image to detect obstructions in power lines."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)