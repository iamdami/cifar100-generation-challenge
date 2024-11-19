import os
import torch
from torchvision import transforms, datasets

def get_dataloader(batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception 모델에 맞게 크기 확장
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
    ])

    train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
