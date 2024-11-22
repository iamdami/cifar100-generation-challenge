import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.dlsm import DLSMGenerator

# Generator 모델 로드
def load_generator(model_path, noise_dim, num_classes, img_channels, device):
    generator = DLSMGenerator(noise_dim, num_classes, img_channels).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

# 이미지 생성
def generate_images(generator, num_samples, noise_dim, num_classes, img_size, device):
    noise = torch.randn(num_samples, noise_dim).to(device)
    labels = torch.randint(0, num_classes, (num_samples,)).to(device)
    with torch.no_grad():
        fake_images = generator(noise, labels).cpu()

    # 이미지 정규화 해제
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Reverse normalization to [0, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Change channel order for matplotlib
    ])
    fake_images = [unnormalize(img) for img in fake_images]
    return fake_images, labels

# 이미지 시각화
def visualize_images(fake_images, labels, num_show=16):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i >= num_show:
            break
        ax.imshow(fake_images[i].clamp(0, 1).numpy())  # Clamp values to valid range
        ax.axis('off')
        ax.set_title(f"Label: {labels[i].item()}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configurations
    MODEL_PATH = "../checkpoints/generator_final.pth"
    NOISE_DIM = 100
    NUM_CLASSES = 100
    IMG_CHANNELS = 3
    NUM_SAMPLES = 16
    IMG_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generator 로드
    generator = load_generator(MODEL_PATH, NOISE_DIM, NUM_CLASSES, IMG_CHANNELS, DEVICE)

    # 이미지 생성
    fake_images, labels = generate_images(generator, NUM_SAMPLES, NOISE_DIM, NUM_CLASSES, IMG_SIZE, DEVICE)

    # 시각화
    visualize_images(fake_images, labels)
