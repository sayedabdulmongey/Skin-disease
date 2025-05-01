import kagglehub

import numpy as np


from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

from transform import *

import re
import json

global_mean, global_std = np.array([0.66200879, 0.49884228, 0.48171022]), np.array([
    0.23122994, 0.19965563, 0.21070801])


def get_data():

    return kagglehub.dataset_download('ismailpromus/skin-diseases-image-dataset')+'/IMG_CLASSES'


def calculate_mean_std():

    global global_mean, global_std

    dataset_path = get_data()

    initial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    raw_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=initial_transform
    )

    raw_data_loader = DataLoader(raw_dataset, batch_size=32, shuffle=False)

    no_images = 0

    for images, _ in raw_data_loader:
        batch_mean = torch.mean(images, dim=(0, 2, 3)).numpy()
        batch_std = torch.std(images, dim=(0, 2, 3)).numpy()

        global_mean += batch_mean*images.size(0)
        global_std += batch_std*images.size(0)

        no_images += images.size(0)

    global_mean /= no_images
    global_std /= no_images


def get_data_loaders(batch_size=32):

    dataset_path = get_data()

    data_transform = get_transforms(global_mean, global_std)

    training_dataset = datasets.ImageFolder(
        dataset_path, transform=data_transform['train'])
    testing_dataset = datasets.ImageFolder(
        dataset_path, transform=data_transform['test'])

    cleaned_class_names = clean_class_names(training_dataset.class_to_idx)

    train_size = int(0.7*len(training_dataset))
    valid_size = int(0.15*len(training_dataset))
    test_size = len(training_dataset) - train_size - valid_size

    generator = torch.Generator().manual_seed(42)
    train_indices, valid_indices, test_indices = random_split(range(
        len(training_dataset)), [train_size, valid_size, test_size], generator=generator)

    train_dataset = torch.utils.data.Subset(
        training_dataset, train_indices.indices)
    valid_dataset = torch.utils.data.Subset(
        training_dataset, valid_indices.indices)
    test_dataset = torch.utils.data.Subset(
        testing_dataset, test_indices.indices)

    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    return data_loaders


def save_class_names(class_names):
    with open('idx_to_class.json', 'w') as f:
        json.dump(class_names, f)


def clean_class_names(class_names):
    cleaned_names = {}
    for class_name, idx in class_names.items():
        match = re.search(r'\d+\.\s*(.+?)(?:\s*[-\d])', class_name)
        if match:
            clean_name = match.group(1).strip()
        else:
            clean_name = class_name  # fallback if regex fails
        cleaned_names[f'{idx}'] = clean_name

    save_class_names(cleaned_names)

    return cleaned_names


if __name__ == '__main__':
    # calculate_mean_std()
    # print(global_mean, global_std)
    data_loaders = get_data_loaders()
    print(data_loaders)
    # dataset_path = get_data()
    # print(dataset_path)
    # dataset = datasets.ImageFolder(root=dataset_path)
    # print(dataset.class_to_idx)
