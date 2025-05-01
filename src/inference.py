import torch

from model import CustomEfficientNet
from transform import get_testing_transforms
from data import global_mean, global_std

import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))


def read_class_names(file_path=f'{dir_path}/idx_to_class.json'):
    with open(file_path, 'r') as f:
        class_names = json.load(f)
    return class_names


def load_model(model_path=f"{dir_path}/model.pth"):

    model = CustomEfficientNet()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image, mean, std):
    transform = get_testing_transforms(mean, std)
    return transform(image)


def prediction(image):

    model = load_model()
    processed_image = preprocess_image(image, global_mean, global_std)
    processed_image = processed_image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(processed_image)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()
        class_names = read_class_names()
        class_name = class_names[f'{class_index}']
        return class_name, 


if __name__ == "__main__":
    path = 'src/model.pth'

    # class_names = read_class_names(path)
    # print(class_names)

    model = load_model(path)
    print(dir_path)
    print("Model loaded successfully.")
