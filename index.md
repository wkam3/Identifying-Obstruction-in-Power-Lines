---
layout: default
title: Computer Vision-Based Obstruction Detection
---

# Computer Vision-Based Obstruction Detection

## Overview
This project leverages advanced computer vision models to automatically detect and classify obstructions near SDG&E power lines. By analyzing visual/aerial data, the system aims to identify potential hazards like vegetation encroachment and damaged equipment before they can cause infrastructure damage or trigger wildfires. This proactive approach enhances grid reliability and public safety through early detection of risks.

## Introduction
In recent years, California has been faced with an intensifying amount of wildfires, largely influenced by urban development and fire-prone regions. A large contributor to these wildfire ignitions is electrical infrastructure when high winds, vegetation, and equipment failures can lead to devastating fires. To mitigate this risk, utility companies such as San Diego Gas & Electric (SDGE) have implemented Public Safety Power Shutoffs (PSPS), a strategy that involves de-energizing power lines during high-risk conditions. While PSPS has proven effective in reducing wildfire ignitions, it also poses far-reaching consequences, including economic losses and health risks for vulnerable populations. As wildfire threats continue to grow, there is a pressing need for proactive, data-driven approaches to assess and mitigate fire risks associated with electrical infrastructure.

A recent study, Utility Pole Fire Risk Inspection from 2D Street-Side Images, presents a computer vision-based framework for identifying wildfire risks associated with utility poles. In their paper, the authors utilize Google Street View imagery to detect poles, vegetation, and evaluate pole inclination, which can indicate structural vulnerability. By leveraging automated image analysis, the study provides a scalable method for prioritizing high-risk infrastructure and informing preventative measures, such as vegetation management and targeted undergrounding of power lines. This approach demonstrates how computer vision techniques can enhance wildfire mitigation efforts by identifying hazards before they lead to catastrophic events. 

This study builds on that approach by integrating computer vision techniques to assess fire risks, focusing on detecting obstructions and identifying vulnerable electrical assets in real-time. The datasets that we are using contain infraction frequencies, outage frequencies, images of electrical assets, and obstructions on those assets. The dataset on infraction frequencies contains a count on all the different types of infractions that SDGE received for each of their electrical assets. With this, we intend to perform exploratory data analysis (EDA) to give us insight into the commonality of infractions. The outage frequencies dataset contains a count on all the different causes of outages with the electrical asset listed at fault. Like the infractions dataset, we will also use this dataset to help give us insight into the commonality of the different types of assets that cause outages. For the images, the electrical assets shown are objects, namely poles and wires, that are commonly dealt with daily. Obstructions on these assets include objects such as balloons and twigs that get commonly caught in the infrastructure. We then use these images to train, test, and validate a model capable of detecting obstructions in electrical assets.

## Methodology
Our methodology for developing an obstruction detection system for electrical assets using YOLOv11 followed a structured pipeline combining computer vision best practices with modern deep learning techniques. YOLOv11, released in late 2024 by Ultralytics, represents the latest advancement in the YOLO (You Only Look Once) family of object detection models. As a Convolutional Neural Network (CNN), it builds upon its predecessors with significant improvements in architecture, efficiency, and accuracy. YOLOv11 introduces innovative features such as the C3k2 (Cross Stage Partial with kernel size 2) block, SPPF (Spatial Pyramid Pooling - Fast), and C2PSA (Convolutional block with Parallel Spatial Attention) components, which enhance feature extraction and improve model accuracy. We chose YOLOv11 for our project due to its superior performance in real-time object detection scenarios, which is crucial for identifying obstructions in poles and wires. YOLOv11 achieves higher mean Average Precision (mAP) on the COCO dataset while using 22% fewer parameters than YOLOv8m, making it computationally efficient without compromising accuracy. This efficiency is particularly beneficial for our application, as it allows for faster processing speeds—approximately 2% quicker than YOLOv10—enabling real-time detection of potential hazards. At its core, YOLOv11 employs a single-stage, anchor-free detection approach within its CNN architecture. It directly predicts bounding boxes and class probabilities for objects in a single forward pass through the neural network. This method eliminates the need for region proposal networks or anchor boxes, resulting in faster inference times and improved handling of objects at various scales. The model's architecture includes an improved backbone and neck design, which enhances its ability to extract relevant features from images, crucial for detecting diverse obstructions in power infrastructure.

**figure here**

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
