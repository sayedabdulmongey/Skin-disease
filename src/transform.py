from torchvision import transforms


def get_transforms(mean, std):
    return {
        'train':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(scale=(0.85, 1.1), translate=(
                0.1, 0.1), degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'test':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    }


def get_testing_transforms(mean, std):
    return get_transforms(mean, std)['test']


def get_training_transforms(mean, std):
    return get_transforms(mean, std)['train']
