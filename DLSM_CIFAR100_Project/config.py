#11.22.2024
CONFIG = {
    "noise_dim": 128,                # 노이즈 차원 증가
    "batch_size": 128,               # 배치 크기 증가 (학습 시간 단축)
    "num_classes": 100,
    "img_channels": 3,
    "lr": 0.0001,                    # 초기 학습률 감소 (안정성 향상)
    "betas": (0.5, 0.999),
    "epochs": 150,                   # 학습 Epoch 제한 (시간 단축)
    "grad_clip": 0.5,                # 그래디언트 클리핑
    "scheduler": "cosine_annealing"  # Cosine Annealing Scheduler 사용
}
