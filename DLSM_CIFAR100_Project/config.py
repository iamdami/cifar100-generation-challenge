#11.25.2024
CONFIG = {
    "noise_dim": 128,                # Dimensionality of noise
    "batch_size": 64,                # Reduced batch size for stable training
    "num_classes": 100,              # Number of classes in CIFAR-100
    "img_channels": 3,               # Image channels (RGB)
    "g_lr": 0.0002,                  # Generator learning rate (increased for faster convergence)
    "d_lr": 0.00005,                 # Discriminator learning rate (decreased for better balance)
    "betas": (0.5, 0.999),           # Beta values for Adam optimizer
    "epochs": 100,                   # Number of training epochs
    "grad_clip": 0.5,                # Gradient clipping to prevent exploding gradients
    "scheduler": "cosine_annealing", # Cosine annealing scheduler for learning rate
    "label_smoothing": 0.9,          # Label smoothing value for discriminator targets
    "noise_reg": 0.1                 # Noise regularization for generator input
}
