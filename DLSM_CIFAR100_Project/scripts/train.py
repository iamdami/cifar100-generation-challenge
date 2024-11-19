import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_dataloader
from models.dlsm import DLSMGenerator, DLSMDiscriminator
from utils.metrics import calculate_fid, calculate_inception_score, calculate_intra_fid
import os
import matplotlib.pyplot as plt

def train_dlsm():
    # Configurations
    noise_dim = 100
    num_classes = 100
    img_channels = 3
    batch_size = 64
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    train_loader, _ = get_dataloader(batch_size, num_workers=8)

    # Initialize Models with DataParallel
    generator = DLSMGenerator(noise_dim, num_classes, img_channels).to(device)
    discriminator = DLSMDiscriminator(num_classes, img_channels).to(device)
    
    if torch.cuda.device_count() > 1:  # 여러 GPU를 사용할 경우
        print(f"Using {torch.cuda.device_count()} GPUs")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Directories for saving results
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "metrics_log.txt"), "w")

    # For plotting
    epoch_list = []
    d_loss_list, g_loss_list = [], []
    fid_list, is_list, intra_fid_list = [], [], []

    # Training Loop
    for epoch in range(epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0
        for real_imgs, labels in train_loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_imgs = generator(noise, labels)

            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            d_loss = criterion(real_validity, torch.ones_like(real_validity)) + \
                     criterion(fake_validity, torch.zeros_like(fake_validity))
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            total_d_loss += d_loss.item()

            # Train Generator
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            total_g_loss += g_loss.item()

        # Average Loss for Epoch
        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)

        # Evaluate Metrics (FID, IS, Intra-FID)
        fid = calculate_fid(generator, train_loader, device, num_samples=5000)
        is_score, is_std = calculate_inception_score(generator, train_loader, device, num_samples=5000)
        intra_fid = calculate_intra_fid(generator, train_loader, device, num_samples=5000)

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

if __name__ == "__main__":
    train_dlsm()
