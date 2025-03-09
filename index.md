---
layout: default
title: Computer Vision-Based Obstruction Detection
---

# Computer Vision-Based Obstruction Detection

## Overview
This project leverages advanced computer vision models to automatically detect and classify obstructions near SDG&E power lines. By analyzing visual/aerial data, the system aims to identify potential hazards like vegetation encroachment and damaged equipment before they can cause infrastructure damage or trigger wildfires. This proactive approach enhances grid reliability and public safety through early detection of risks.

**Video of working model demonstration**

## Introduction
California has been facing surges in wildfires, often sparked by electrical infrastructure during high winds and equipment failures. Currently, SDG&E uses Public Safety Power Shutoffs (PSPS), cutting power during dangerous conditions, to reduce the spread of these fires. While effective, PSPS also causes economic losses and health risks. As wildfire threats grow, data-driven strategies to mitigate these risks proactively are essential to maintain the safety and well-being of San Diegans.

## Methodology
Our methodology for developing an obstruction detection system for electrical assets using YOLOv11 followed a structured pipeline combining computer vision best practices with modern deep learning techniques. YOLOv11, released in late 2024 by Ultralytics, represents the latest advancement in the YOLO (You Only Look Once) family of object detection models. We chose YOLOv11 for our project due to its superior performance in real-time object detection scenarios. YOLOv11 achieves higher mean Average Precision (mAP) and is more computationally efficient than its predecessors allowing for quicker and more accurate identification.

**figure here**

<style>
  details {
    border: 2px solid #007acc; /* Blue outline */
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    background-color: #f9f9f9;
    cursor: pointer;
  }

  summary {
    font-weight: bold;
    cursor: pointer;
    padding: 5px;
  }

  details[open] {
    background-color: #e6f2ff; /* Light blue when expanded */
  }

  details p {
    margin-top: 10px;
    padding-left: 5px;
  }
</style>

<details>
  <summary>🔍 <strong>Exploratory Data Analysis (EDA)</strong></summary>
  <p>We began by conducting EDA on outage data and infraction data per equipment type...</p>
</details>

<details>
  <summary>📊 <strong>Dataset Curation</strong></summary>
  <p>We gathered relevant videos and images, then used Roboflow for image annotation...</p>
</details>

<details>
  <summary>🖼 <strong>Image Annotation & Augmentation</strong></summary>
  <p>We labeled images with bounding boxes for obstructions and used augmentation techniques...</p>
</details>

The implementation of our models' consisted of five primary phases: EDA(exploratory data analysis), dataset curation, model configuration, training optimization, and performance evaluation. We first began by conducting EDA on outage data and infraction data per equipment type and we able to see that poles,conductors, and crossarms were the electrical assets that had the highest number of outages and infractions. We then began curating data for our model by gathering relevant videos and images then pipelining our images into a tool called Roboflow where we conducted image annotation. Yolo datasets comprise of both images and annotations as an additional json file. To create these annotations we drew  bounding polygons for each of our images for each class we were labeling (poles, wires, obstructions). Roboflow provided user interface to perform these annotations as well as transforming our annotations into an accompanying json label dataset. We ended up compiling a dataset of 383 annotated images with an almost equal split between obstruction and non obstruction images. After creating our dataset we also included preprocessing steps as well as augmentations to generate more images. In our preprocessing we included auto-orient, resize, and contrast adjustment. Moreover, in our augmentation steps we performed both horizontal and vertical flips, rotations, shears, changes in hue, brightness, and exposure, blur, and noise. This resulted in an increase from 383 images to 1023 images. We then split our data into 94% training, 3% validation, and 3% test data. To train our model we took three different approaches. Our first approach was training directly on all of our labeled classes: poles, wires, and obstructions. Our second approach sought to use three different models for each respective class. These three models were independently trained on our full dataset to ensure optimal performance in each category. Following the training phase we applied each models' predictions to our test set and then implemented a NMS (Non-maximum suppression) strategy to refine our results and isolate our most confident predictions. This technique allowed us to retain only the most robust detections. Our final approach was training only on obstructions. We utilized a new dataset of only 81 images   The parameters we used to train were 50 epochs, input image size of 416x416 pixels, batch sizes of 4, Automatic Mixed Precision enabled, and classes set to obstruction only. Following training we developed code to annotate our test images with our models' predictions and display metrics.

    Notes:
   Train model with augmentation

## Results
Our investigation into obstruction detection for electrical assets using YOLOv11 yielded insightful results across three distinct modeling approaches. Each approach offered unique perspectives on the challenge of identifying obstructions in complex electrical infrastructure environments.

**Model Performance Comparison table**

**Obstruction-Only Model**
The obstruction-only model emerged as our best-performing approach. By focusing exclusively on detecting obstructions, this model demonstrated superior performance in identifying potential hazards.

The high precision (0.9762) indicates that when the model identified an obstruction, it was correct 97.62% of the time. While the recall (0.5714) suggests room for improvement in detecting all obstructions, the model's ability to accurately identify obstructions when it did detect them was noteworthy.

**Multi-Class Model (Poles, Wires, Obstructions)**
Our multi-class model, trained to simultaneously detect poles, wires, and obstructions, showed lower performance compared to the obstruction-only model. While this approach provided a comprehensive view of the electrical infrastructure, it struggled with the complexity of distinguishing between multiple classes in often cluttered environments.

**Independent Models for Each Class**
Our approach using separate models for poles, wires, and obstructions, followed by NMS, yielded results that fell between the other two approaches. This method showed improvements over the multi-class model but still fell short of the obstruction-only model's performance.

## Conclusion
Our findings underscore the importance of tailoring the modeling approach to the specific needs of the task at hand. In the context of electrical asset management, the ability to reliably detect obstructions proved more valuable than a more comprehensive but potentially less accurate multi-class detection system.

## Data Sources
- SDG&E Infraction Reports (2020-2024)
- Equipment maintenance records
- Risk assessment documentation
- Google images
