import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_dataloader
from models.dlsm import DLSMGenerator, DLSMDiscriminator
from utils.metrics import calculate_fid, calculate_inception_score, calculate_intra_fid
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("/home/cipher/Dami/DLSM_CIFAR100_Project/"))

# Config
CONFIG = {
    "noise_dim": 128,                # 노이즈 차원 증가
    "batch_size": 128,               # 배치 크기 증가 (학습 시간 단축)
    "num_classes": 100,
    "img_channels": 3,
    "lr": 0.0001,                    # 초기 학습률 감소 (안정성 향상)
    "betas": (0.5, 0.999),
    "epochs": 150,                   # 학습 Epoch 제한
    "grad_clip": 0.5,                # 그래디언트 클리핑
    "scheduler": "cosine_annealing"  # Cosine Annealing Scheduler 사용
}

# Plotting Function
def plot_results(epochs, d_loss, g_loss, fid, is_score, intra_fid, save_dir):
    plt.figure()
    # Loss Plot
    plt.plot(epochs, d_loss, label="Discriminator Loss")
    plt.plot(epochs, g_loss, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    # FID, IS, Intra-FID Plot
    plt.figure()
    plt.plot(epochs, fid, label="FID")
    plt.plot(epochs, is_score, label="Inception Score")
    plt.plot(epochs, intra_fid, label="Intra-FID")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("FID, IS, and Intra-FID Scores")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
    plt.close()


# Training Function
def train_dlsm():
    # Unpack Config
    noise_dim = CONFIG["noise_dim"]
    batch_size = CONFIG["batch_size"]
    lr = CONFIG["lr"]
    epochs = CONFIG["epochs"]
    grad_clip = CONFIG["grad_clip"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    train_loader, _ = get_dataloader(batch_size, num_workers=8)

    # Initialize Models
    generator = DLSMGenerator(noise_dim, CONFIG["num_classes"], CONFIG["img_channels"]).to(device)
    discriminator = DLSMDiscriminator(CONFIG["num_classes"], CONFIG["img_channels"]).to(device)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=CONFIG["betas"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=CONFIG["betas"])

    # Scheduler
    if CONFIG["scheduler"] == "cosine_annealing":
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs)
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs)

    # Logging Setup
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "metrics_log.txt"), "w")

    # For plotting
    epoch_list = []
    d_loss_list, g_loss_list = [], []
    fid_list, is_list, intra_fid_list = [], [], []

    # Training Loop
    for epoch in range(epochs):
        total_d_loss, total_g_loss = 0.0, 0.0
        generator.train()
        discriminator.train()

        for real_imgs, labels in train_loader:
            # Ensure batch sizes match
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            current_batch_size = real_imgs.size(0)  # Handle last batch

            # Generate Fake Images
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_imgs = generator(noise, labels)
            
            # Debugging output (print every 10th batch only to reduce log size)
            if epoch % 10 == 0:
                print(f"Noise shape: {noise.shape}, Labels shape: {labels.shape}")
                print(f"Generated Fake Images shape: {fake_imgs.shape}")

            # Train Discriminator
            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            d_loss = criterion(real_validity, torch.ones_like(real_validity)) + \
                     criterion(fake_validity, torch.zeros_like(fake_validity))
            optimizer_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)  # Gradient Clipping
            optimizer_d.step()
            total_d_loss += d_loss.item()

            # Train Generator
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
            optimizer_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)  # Gradient Clipping
            optimizer_g.step()
            total_g_loss += g_loss.item()

        # Scheduler Update
        if CONFIG["scheduler"] == "cosine_annealing":
            scheduler_g.step()
            scheduler_d.step()

        # Average Loss for Epoch
        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)

        # Evaluate Metrics
        fid = calculate_fid(generator, train_loader, device, num_samples=1000)
        is_score, is_std = calculate_inception_score(generator, train_loader, device, num_samples=1000)
        intra_fid = calculate_intra_fid(generator, train_loader, device, num_samples=1000)

        # Log Epoch Results
        epoch_list.append(epoch + 1)
        d_loss_list.append(avg_d_loss)
        g_loss_list.append(avg_g_loss)
        fid_list.append(fid)
        is_list.append(is_score)
        intra_fid_list.append(intra_fid)

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | FID: {fid:.4f} | IS: {is_score:.4f} ± {is_std:.4f} | Intra-FID: {intra_fid:.4f}")
        log_file.write(f"{epoch+1},{avg_d_loss:.4f},{avg_g_loss:.4f},{fid:.4f},{is_score:.4f},{is_std:.4f},{intra_fid:.4f}\n")

    log_file.close()
    print("Training Complete!")

    # Plot results
    plot_results(epoch_list, d_loss_list, g_loss_list, fid_list, is_list, intra_fid_list, log_dir)


if __name__ == "__main__":
    train_dlsm()
