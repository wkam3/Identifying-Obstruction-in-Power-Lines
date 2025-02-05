#!/usr/bin/env python

"python script.py setup data train annotate evaluate display"
import sys
import json

from functions import (
    setup_environment,
    initialize_roboflow,
    download_dataset,
    train_model,
    annotate_images,
    evaluate_model,
    display_results
)

def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def main(targets):
    config = load_config()

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

    if 'train' in targets:
        model = train_model(
            model_path=config['model']['path'],
            data_config=config['model']['data_config'],
            **config['model']['training']
        )
        print("model trained")

    if 'annotate' in targets:
        annotate_images(
            model=model,
            image_folder=config['image_processing']['input_folder'],
            output_folder=config['image_processing']['output_folder'],
            class_colors={int(k): tuple(v) for k, v in config['class_colors'].items()},
            class_names={int(k): v for k, v in config['class_names'].items()}
        )
        print("images annotated stored here /content/Annotated/ ")

    if 'evaluate' in targets:
        metrics = evaluate_model(model)
        print("Validation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    if 'display' in targets:
        display_results(config['image_processing']['output_folder'])

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
