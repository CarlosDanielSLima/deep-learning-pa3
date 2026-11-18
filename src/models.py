from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import copy

import copy
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from tqdm import tqdm

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


class MultiCropDataAugmentation:
    def __init__(self, size_crops, num_crops, min_scale, max_scale):
        self.size_crops = size_crops
        self.num_crops = num_crops
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.transforms = []
        for i in range(len(size_crops)):
            # Cria um pipeline de transformação para cada resolução de corte
            random_resized_crop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale[i], max_scale[i])
            )
            # Aumentos de dados padrão do SwAV (blur com kernel pequeno, pois as imagens são 32x32 e 16x16)
            self.transforms.append(transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                transforms.ToTensor(),
                # Normalização com as médias e desvios padrão do ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

    def __call__(self, x):
        # Aplica as transformações para gerar todas as visualizações
        all_crops = []
        # Itera dinamicamente sobre a quantidade de escalas fornecidas
        for i, num in enumerate(self.num_crops):
            for _ in range(num):
                all_crops.append(self.transforms[i](x))
        return all_crops


class SwAVModel(nn.Module):
    # in_features=512 corresponde à saída da layer4 da ResNet-18
    def __init__(self, in_features=512, dense_1=1024, dense_2=96, prototype_dim=300, backbone: nn.Module=None):
        super().__init__()

        # --- Parte do Backbone (ResNet-18) ---
        if backbone is None:
            self.backbone = build_resnet18()
        else:
            self.backbone = backbone
        
        # Romove a camada fc
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # --- Parte da Cabeça de Projeção e Protótipo ---
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, dense_1),
            nn.BatchNorm1d(dense_1),
            nn.ReLU(),
            nn.Linear(dense_1, dense_2)
        )
        self.prototype = nn.Linear(dense_2, prototype_dim, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        # Para depurar, você pode descomentar a linha abaixo e verificar a dimensão:
        # print(f"Shape do vetor de características: {features.shape}") # Deve ser [batch_size, 128]

        projection = self.projection_head(features)
        projection_normalized = F.normalize(projection, dim=1, p=2)
        prototype_output = self.prototype(projection_normalized)

        return features, projection_normalized, prototype_output


@torch.no_grad()
def sinkhorn(sample_prototype_batch, epsilon=0.05, n_iters=3, device="cpu"):
    Q = torch.exp(sample_prototype_batch / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q

    r = torch.ones(K, device=device) / K
    c = torch.ones(B, device=device) / B

    for _ in range(n_iters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

    final_Q = Q / torch.sum(Q, dim=0, keepdim=True)
    return final_Q.t()


def train_step(
    input_views, 
    swav_model: SwAVModel, 
    optimizer, 
    crops_for_assign, 
    temperature,
    device="cpu"
):
    """
    Função de treinamento para um único passo (batch), com múltiplas resoluções.
    """
    # Normaliza os protótipos a cada passo (projeção na esfera unitária)
    with torch.no_grad():
        w = F.normalize(swav_model.prototype.weight.data.clone(), dim=1, p=2)
        swav_model.prototype.weight.copy_(w)

    # Listas para armazenar as saídas de cada forward pass
    all_projections = []
    all_prototypes = []

    # === FORWARD PASS SEPARADO PARA CADA RESOLUÇÃO ===
    # Itera sobre cada tensor de visualização (cada um contendo um batch de uma resolução específica)
    for view in input_views:
        view = view.to(device)

        # Realiza o forward pass para esta visualização específica
        features, projections, prototypes = swav_model(view)

        all_projections.append(projections)
        all_prototypes.append(prototypes)

    # Concatena as saídas DEPOIS do forward pass
    prototypes = torch.cat(all_prototypes, dim=0)

    # Separa os protótipos para cada visualização
    batch_size = input_views[0].shape[0]
    prototypes_split = torch.split(prototypes, batch_size, dim=0)

    # ============ cálculo da perda swav ============
    loss = 0
    for i, crop_id in enumerate(crops_for_assign):
        with torch.no_grad():
            out = prototypes_split[crop_id]
            q = sinkhorn(out, device=device)

        # Previsão cruzada: todas as outras visualizações preveem o código q
        subloss = 0
        for v in range(len(input_views)):
            if v == crop_id:
                continue
            log_p = F.log_softmax(prototypes_split[v] / temperature, dim=1)
            subloss -= torch.mean(torch.sum(q * log_p, dim=1))

        loss += subloss / (len(input_views) - 1)

    loss /= len(crops_for_assign)

    # ============ backprop ... ============
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_swav(
    swav_model: SwAVModel, 
    dataloader, 
    optimizer,
    crops_for_assign, 
    temperature, 
    epochs=50,
    device="cpu"
):
    """
    Loop de treinamento principal por épocas, adaptado para o SwAVModel unificado.
    """
    swav_model.to(device)
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in range(epochs):
        swav_model.train() # Coloca o modelo em modo de treinamento

        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for (inputs, _) in pbar:
            # inputs é uma lista de tensores (um para cada visualização/crop)
            loss = train_step(inputs, swav_model, optimizer,
                                      crops_for_assign, temperature, device=device)
            step_wise_loss.append(loss)
            epoch_loss += loss
            pbar.set_postfix(loss=f'{loss:.3f}')

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_wise_loss.append(avg_epoch_loss)
        print(f"Epoch: {epoch + 1} | Loss: {avg_epoch_loss:.3f}")

    return epoch_wise_loss, swav_model


class LinearProbe(nn.Module):
    # num_classes = 21 (assumindo 105 imagens no total com 5 por classe)
    def __init__(self, backbone, num_classes=21):
        super().__init__()
        
        self.backbone = backbone
        
        # 1. Congela TODOS os pesos do backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Cria apenas a cabeça de classificação, que será treinada
        self.fc = nn.Linear(512, num_classes)
        
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        # Garante que camadas de Dropout/BatchNorm não alterem o embedding
        self.backbone.eval() 
        
        with torch.no_grad():
            features = self.backbone(x)
            features = torch.flatten(features, 1)
            
        logits = self.fc(features)
        return logits


def train_linear_probe(
    probe_model: LinearProbe, 
    train_loader_105, 
    val_loader_public, 
    epochs=50, 
    lr=1e-3, 
    device="cpu"
):
    probe_model.to(device)
    
    # Otimizador atualiza APENAS os pesos da camada linear
    optimizer = torch.optim.AdamW(probe_model.fc.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_weights = copy.deepcopy(probe_model.state_dict())
    
    for epoch in range(epochs):
        # === TREINO ===
        probe_model.train() # Coloca o FC em train (o backbone é mantido em eval internamente no forward)
        
        for images, labels in train_loader_105:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
    # === AVALIAÇÃO NO TEST_PUBLIC (Fim do Treino) ===
    probe_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader_public:
            images, labels = images.to(device), labels.to(device)
            outputs = probe_model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Cálculo das métricas
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = acc
            best_weights = copy.deepcopy(probe_model.state_dict())
            
    # Restaura os pesos da melhor época de validação!
    probe_model.load_state_dict(best_weights)
    return best_val_f1, best_val_acc


class FineTunedModel(nn.Module):
    def __init__(self, backbone, num_classes=21, dropout_p=0.3):
        super().__init__()
        
        self.backbone = backbone
        
        # 1. Destravar TODOS os pesos do backbone para o fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # 2. Criar a nova cabeça de classificação
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), # Dropout para evitar overfitting
            nn.Linear(512, num_classes) # 512 é a saída da ResNet-18
        )
        
        # Inicialização dos pesos da camada linear
        self.fc[1].weight.data.normal_(mean=0.0, std=0.01)
        self.fc[1].bias.data.zero_()

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

def train_fine_tuning(
    model: FineTunedModel, 
    train_loader_105, 
    val_loader_public, 
    epochs=40, 
    device="cpu"
):
    model.to(device)
    
    # Taxas de aprendizagem separadas. 
    # O backbone aprende devagar (1e-5), a nova cabeça aprende rápido (1e-3).
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    melhor_f1 = -1.0
    melhor_acc = 0.0
    melhor_epoca = -1
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # === TREINO ===
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader_105:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        scheduler.step()
            
        # === AVALIAÇÃO ===
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader_public:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calcula as métricas no fim de cada época
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        loss_media = epoch_loss / len(train_loader_105)
        
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_acc = acc
            melhor_epoca = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Nova melhor F1 Macro: {melhor_f1:.4f} na época {melhor_epoca} | Val Acc: {melhor_acc:.4f}")
            
        # Imprime o progresso a cada 10 épocas ou na última
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} | Loss Treino: {loss_media:.4f} | Val F1: {f1:.4f} | Val Acc: {acc:.4f}")
        
    print(f"--- Fim do Fine-Tuning. Melhor F1 Macro: {melhor_f1:.4f} ---")

    model.load_state_dict(best_weights)
    # Retorna o modelo treinado, o último F1 e a última Accuracy
    return model, melhor_f1, melhor_acc


class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar a ativação e o gradiente da camada alvo
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward
        output = self.model(input_tensor)
        
        # Cria um tensor one-hot para a classe alvo e faz o backward
        target = torch.zeros_like(output)
        target[0][target_class] = 1
        output.backward(gradient=target, retain_graph=True)
        
        # Calcula os pesos do Grad-CAM (média espacial dos gradientes)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Multiplica as ativações pelos pesos
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam) # Apenas influências positivas
        
        # Normaliza o CAM para [0, 1]
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().cpu().detach().numpy()



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