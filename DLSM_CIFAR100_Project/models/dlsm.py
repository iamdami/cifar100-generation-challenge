import torch
import torch.nn as nn
import torch.nn.functional as F


class DLSMGenerator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels):
        super(DLSMGenerator, self).__init__()
        self.noise_dim = noise_dim  # noise_dim 속성 추가
        self.num_classes = num_classes  # num_classes 속성 추가
        self.label_embed = nn.Embedding(num_classes, noise_dim)

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
        input_data = torch.cat((noise, label_encoding), dim=1)
        output = self.model(input_data).view(-1, 3, 32, 32)
        return output


class DLSMDiscriminator(nn.Module):
    def __init__(self, num_classes, img_channels):
        super(DLSMDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=2, padding=1),  # (32x32) -> (16x16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (16x16) -> (8x8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (8x8) -> (4x4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (4x4) -> (2x2)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((2, 2))  # 출력 크기를 2x2로 강제
        )
        self.flatten = nn.Flatten()

        # 최종 Conv2D 출력 크기 계산: 512 채널, 2x2 크기
        self.fc_input_dim = 512 * 2 * 2  # = 2048

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim + num_classes, 1024),  # num_classes 추가
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_features = self.model(img)
        img_features = self.flatten(img_features)
    
        labels_one_hot = F.one_hot(labels, num_classes=100).float().to(img.device)
        input_data = torch.cat((img_features, labels_one_hot), dim=1)
        output = self.fc(input_data)
        return output
