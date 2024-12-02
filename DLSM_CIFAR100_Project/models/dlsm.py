import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreModel(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels):
        super(ScoreModel, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes

        self.label_embed = nn.Embedding(num_classes, noise_dim)

        self.fc1 = nn.Linear(noise_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, img_channels * 32 * 32)  # Output flattened image size

    def forward(self, noise, labels):
        # Get label embeddings
        label_encoding = self.label_embed(labels)
        
        # Ensure label_encoding has shape (batch_size, noise_dim)
        if label_encoding.dim() > 2:
            label_encoding = label_encoding.squeeze(1)  # Remove unnecessary dimensions
        label_encoding = label_encoding.view(label_encoding.size(0), self.noise_dim)  # Flatten if necessary
        
        # Concatenate noise and label encoding
        x = torch.cat([noise, label_encoding], dim=1)

        # Pass through fully connected layers
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # Reshape to (batch_size, 3, 32, 32) for image output
        out = out.view(out.size(0), 3, 32, 32)
        return out


class ScoreDiscriminator(nn.Module):
    def __init__(self, num_classes, img_channels):
        super(ScoreDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 4 * 4 + num_classes, 1)

    def forward(self, img, labels):
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten before passing to fully connected layer

        # One-hot encode the labels and concatenate with image features
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(img.device)
        x = torch.cat((x, labels_one_hot), dim=1)

        # Fully connected layer
        score = self.fc(x)
        return score
