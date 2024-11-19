import torch
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.functional as TF
from scipy.linalg import sqrtm

# Load Inception v3 model
def get_inception_model(device):
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()
    return inception_model

# Get activations for FID
def get_activations(data_loader, model, device, num_samples):
    model.eval()
    activations = []

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i * images.size(0) >= num_samples:
                break
            images = torch.stack([TF.resize(img.cpu(), (75, 75)) for img in images]).to(device)
            pred = model(images)
            activations.append(pred.cpu().numpy())

    return np.concatenate(activations, axis=0)

# FID Calculation
def calculate_fid(generator, data_loader, device, num_samples=5000):
    inception_model = get_inception_model(device)
    real_activations = get_activations(data_loader, inception_model, device, num_samples)

    generator.eval()
    fake_images = []
    with torch.no_grad():
        for _ in range(num_samples // data_loader.batch_size):
            noise = torch.randn(data_loader.batch_size, generator.noise_dim).to(device)
            labels = torch.randint(0, generator.num_classes, (data_loader.batch_size,)).to(device)
            fake_images.append(generator(noise, labels).cpu())
    fake_images = torch.cat([TF.resize(img, (75, 75)).unsqueeze(0) for img in torch.cat(fake_images)])
    fake_activations = get_activations([(fake_images, None)], inception_model, device, num_samples)

    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return np.real(fid)

# Inception Score Calculation
def calculate_inception_score(generator, data_loader, device, num_samples=5000):
    """
    Calculate the Inception Score (IS) for generated images.
    """
    inception_model = get_inception_model(device)
    generator.eval()
    probs = []
    with torch.no_grad():
        for _ in range(num_samples // data_loader.batch_size):
            noise = torch.randn(data_loader.batch_size, generator.noise_dim).to(device)
            labels = torch.randint(0, generator.num_classes, (data_loader.batch_size,)).to(device)
            fake_images = generator(noise, labels)

            # Resize fake images to 75x75
            fake_images = torch.stack([TF.resize(img, (75, 75)) for img in fake_images])

            prob = inception_model(fake_images.to(device))
            probs.append(prob.cpu().numpy())

    probs = np.concatenate(probs, axis=0)

    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    kl_div = probs * (np.log(probs) - np.log(np.expand_dims(np.mean(probs, axis=0), 0)))
    is_score = np.exp(np.mean(np.sum(kl_div, axis=1)))
    return is_score, np.std(np.sum(kl_div, axis=1))


# Intra-FID Calculation
def calculate_intra_fid(generator, data_loader, device, num_samples=5000):
    """
    Calculate the Intra-FID for real and generated images grouped by class.
    """
    inception_model = get_inception_model(device)
    generator.eval()
    real_activations_by_class = {c: [] for c in range(generator.num_classes)}
    fake_activations_by_class = {c: [] for c in range(generator.num_classes)}

    with torch.no_grad():
        for real_images, labels in data_loader:
            real_images, labels = real_images.to(device), labels.to(device)
            real_activations = inception_model(real_images).cpu().numpy()
            for act, label in zip(real_activations, labels.cpu().numpy()):
                real_activations_by_class[label].append(act)

        # Initialize fake_images list
        fake_images = []
        for _ in range(num_samples // data_loader.batch_size):
            noise = torch.randn(data_loader.batch_size, generator.noise_dim).to(device)
            labels = torch.randint(0, generator.num_classes, (data_loader.batch_size,)).to(device)
            generated_images = generator(noise, labels)
            fake_images.extend(generated_images)

        # Resize fake images to the required dimensions
        fake_images = torch.stack([TF.resize(img, (75, 75)) for img in fake_images])

        # Calculate activations
        fake_activations = inception_model(fake_images).cpu().numpy()
        for act, label in zip(fake_activations, labels.cpu().numpy()):
            fake_activations_by_class[label].append(act)

    intra_fid = 0
    for c in range(generator.num_classes):
        real_acts = np.array(real_activations_by_class[c])
        fake_acts = np.array(fake_activations_by_class[c])
        if len(real_acts) == 0 or len(fake_acts) == 0:
            continue

        mu_real, sigma_real = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_acts, axis=0), np.cov(fake_acts, rowvar=False)

        diff = mu_real - mu_fake
        covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
        intra_fid += np.real(np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean))

    intra_fid /= generator.num_classes  # Average across classes
    return intra_fid
