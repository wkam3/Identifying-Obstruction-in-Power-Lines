#!/usr/bin/env python

"python script.py AllClasses setup data train annotate evaluate"
"python script.py NMS setup data train_nms_model annotate_nms_model evaluate_nms_model"
import sys
import json
import os
from ultralytics import YOLO

from functions import (
    setup_environment,
    initialize_roboflow,
    download_dataset,
    train_model,
    annotate_images,
    evaluate_model,
    display_results,
    train_nms_model,
    apply_nms,
    iou,
    annotate_images_NMS,
    evaluate_model_nms
    
)
home_dir = os.path.expanduser("~")
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def main(targets):

    # Determine which config to use based on user input
    if 'AllClasses' in targets:
        config_path = os.path.join(home_dir, "Identifying-Obstruction-in-Power-Lines", "Project_code", "config_AllClasses.json")
    elif 'ObstructionsOnly' in targets:
        config_path =  os.path.join(home_dir, "Identifying-Obstruction-in-Power-Lines", "Project_code", "config_ObstructionsOnly.json")
    elif 'NMS' in targets:
        config_path =  os.path.join(home_dir, "Identifying-Obstruction-in-Power-Lines", "Project_code", "config_NMS.json")
    else:
        print("Please specify a valid model type: poles, wires, obstructions, or all")
        return

    config = load_config(config_path)

    if 'setup' in targets:
        setup_environment()

    if 'data' in targets:
        rf = initialize_roboflow(config['api_key'])
        dataset = download_dataset(
            rf,
            config['roboflow']['workspace'],
            config['roboflow']['project'],
            config['roboflow']['version']
        )
        print(f"Dataset downloaded")

    if 'train_nms_model' in targets:
        models = train_nms_model(
            model_path=config['model']['path'],
            **config['model']['training']
        )
        print("model trained")
    if 'annotate_nms_model' in targets:
        annotate_images_NMS(
            models=models,
            class_colors={k: tuple(v) for k, v in config['class_colors'].items()}
        )
        print("images annotated stored in annotated folder ")

    if 'evaluate_nms_model' in targets:
        metrics = evaluate_model_nms(models)


    if 'train' in targets:
        model = train_model(
            model_path=config['model']['path'],
            **config['model']['training']
        )
        print("model trained")

    if 'annotate' in targets:
        annotate_images(
            model=model,
            class_colors={int(k): tuple(v) for k, v in config['class_colors'].items()},
            class_names={int(k): v for k, v in config['class_names'].items()}
        )
        print("images annotated stored in annotated folder ")

    if 'evaluate' in targets:
        evaluate_model(model) 

    if 'display' in targets:
        display_results(config['image_processing']['output_folder'])

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
