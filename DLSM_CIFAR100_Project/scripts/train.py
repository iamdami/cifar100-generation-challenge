import os
import sys
sys.path.append(os.path.abspath("/home/cipher/Dami/DLSM_CIFAR100_Project"))
import torch
import torch.nn as nn
import torch.optim as optim
from models.dlsm import ResNetGenerator, ResNetDiscriminator
from data.data_loader import get_dataloader
from utils.metrics import calculate_fid, calculate_inception_score, calculate_intra_fid
import matplotlib.pyplot as plt

CONFIG = {
    "noise_dim": 128,
    "batch_size": 64,  # Batch size set to 64 as per the experiment
    "num_classes": 100,
    "img_channels": 3,
    "lr": 0.0002,  # Learning rate for both models
    "epochs": 100,  # Number of epochs
    "grad_clip": 0.5,  # Gradient clipping value
    "fid_sample_size": 1000,  # Number of samples for FID calculation
}

def train_dlsm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DataLoader with normalization to [-1, 1]
    train_loader, _ = get_dataloader(CONFIG["batch_size"])

    generator = ResNetGenerator(CONFIG["noise_dim"], CONFIG["num_classes"], CONFIG["img_channels"]).to(device)
    discriminator = ResNetDiscriminator(CONFIG["num_classes"], CONFIG["img_channels"]).to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=CONFIG["lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=CONFIG["lr"])

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    fid_list, is_list, intra_fid_list = [], [], []

    for epoch in range(CONFIG["epochs"]):
        generator.train()
        discriminator.train()

        total_d_loss, total_g_loss = 0, 0

        for real_imgs, labels in train_loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            current_batch_size = real_imgs.size(0)

            # Noise generation for the generator
            noise = torch.randn(current_batch_size, CONFIG["noise_dim"]).to(device)
            fake_imgs = generator(noise, labels)

            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            real_labels = torch.full_like(real_validity, 0.9)
            fake_labels = torch.zeros_like(fake_validity)

            # Discriminator loss
            d_loss = criterion(real_validity, real_labels) + criterion(fake_validity, fake_labels)
            optimizer_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), CONFIG["grad_clip"])
            optimizer_d.step()

            # Generator loss
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
            optimizer_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), CONFIG["grad_clip"])
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        # Calculate metrics for this epoch
        fid = calculate_fid(generator, train_loader, device, CONFIG["fid_sample_size"])
        is_score, _ = calculate_inception_score(generator, train_loader, device, CONFIG["fid_sample_size"])
        intra_fid = calculate_intra_fid(generator, train_loader, device, CONFIG["fid_sample_size"])

        fid_list.append(fid)
        is_list.append(is_score)
        intra_fid_list.append(intra_fid)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f} | FID: {fid:.4f} | IS: {is_score:.4f} | Intra-FID: {intra_fid:.4f}")

    # Plot metrics
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, CONFIG["epochs"]+1), fid_list, label="FID")
    plt.plot(range(1, CONFIG["epochs"]+1), is_list, label="Inception Score")
    plt.plot(range(1, CONFIG["epochs"]+1), intra_fid_list, label="Intra-FID")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "metrics_plot.png"))
    plt.close()

if __name__ == "__main__":
    train_dlsm()
