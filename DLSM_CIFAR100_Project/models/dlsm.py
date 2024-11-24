import torch
import torch.nn as nn
import torch.nn.functional as F


class DLSMGenerator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels):
        super(DLSMGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, noise_dim)  # Correct embedding size

        self.model = nn.Sequential(
            nn.Linear(noise_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 32 * 32),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_encoding = self.label_embed(labels)
        print(f"Noise shape: {noise.shape}, Label Encoding shape: {label_encoding.shape}")  # Debugging
    
        # Match the batch size if labels have fewer samples
        if label_encoding.size(0) != noise.size(0):
            raise ValueError(f"Batch size mismatch: noise {noise.size(0)}, labels {label_encoding.size(0)}")
    
        input_data = torch.cat((noise, label_encoding), dim=1)
        output = self.model(input_data).view(-1, 3, 32, 32)
        return output


class DLSMDiscriminator(nn.Module):
    def __init__(self, num_classes, img_channels):
        super(DLSMDiscriminator, self).__init__()
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((2, 2))  # 2x2 출력 강제
        )
        self.flatten = nn.Flatten()

        self.fc_input_dim = 512 * 2 * 2  # Flatten 이후 크기
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Feature extraction
        img_features = self.model(img)
        img_features = self.flatten(img_features)

        # One-hot encode labels
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(img.device)
        print(f"Image Features shape: {img_features.shape}, Label One-hot shape: {labels_one_hot.shape}")
        
        # Concatenate features with labels
        input_data = torch.cat((img_features, labels_one_hot), dim=1)

        # Final prediction
        output = self.fc(input_data)
        return output
