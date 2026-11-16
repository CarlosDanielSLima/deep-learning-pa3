import torch
import torch.nn as nn
from torchvision import models

MAX_ALLOWED_PARAMS = 12_000_000


def count_parameters(model: nn.Module) -> int:
    """Conta o total de parâmetros do modelo conforme regra do PA3."""
    return sum(p.numel() for p in model.parameters())


class SmallCNN(nn.Module):
    """
    Parte 0 (1a): CNN pequena treinada do zero.
    4 blocos convolucionais (32 -> 64 -> 128 -> 256) + Global Average Pooling + Linear.
    """
    def __init__(self, num_classes: int = 21, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: (3, 256, 352) -> (32, 128, 176)
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 2: (32, 128, 176) -> (64, 64, 88)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 3: (64, 64, 88) -> (128, 32, 44)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 4: (128, 32, 44) -> (256, 16, 22)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Global Average Pooling: (256, 16, 22) -> (256, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_resnet18(num_classes: int = 21, pretrained: bool = True, dropout_p: float = 0.2) -> nn.Module:
    """
    Parte 0 (1b) / Parte 1 / Parte 2: Backbone ResNet-18.
    Substitui a camada final (fc) para 21 classes.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features  # 512 na ResNet-18
    if dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
        
    return model


def build_model(model_type: str = "resnet18", num_classes: int = 21, pretrained: bool = True) -> nn.Module:
    """
    Factory de modelos com verificação automática do limite de 12M de parâmetros.
    model_type: 'small_cnn' ou 'resnet18'
    """
    if model_type == "small_cnn":
        model = SmallCNN(num_classes=num_classes)
    elif model_type == "resnet18":
        model = build_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    n_params = count_parameters(model)
    assert n_params <= MAX_ALLOWED_PARAMS, (
        f"ERRO: Modelo {model_type} tem {n_params:,} parâmetros (limite: {MAX_ALLOWED_PARAMS:,})!"
    )
    return model