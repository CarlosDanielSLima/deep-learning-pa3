import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from src.data_loader import set_seed


def create_optimizer(
    model: nn.Module,
    model_type: str = "resnet18",
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-2,
):
    """
    Configura o otimizador AdamW.
    - Para resnet18 (fine-tuning): usa discriminative learning rates (menor no backbone, maior na fc).
    - Para small_cnn (do zero): usa lr_head único para toda a rede.
    """
    if model_type == "resnet18":
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
        head_params     = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=weight_decay)
    
    return optimizer


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_classes: int = 21):
    """
    Avalia o modelo em um DataLoader (val ou test_public) sem calcular gradientes.
    Retorna dicionário com loss, acc, f1_macro, f1_per_class, y_true e y_pred.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    model_type: str = "resnet18",
    epochs: int = 40,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-2,
    save_path: str = "checkpoints/best_model.pt",
    seed: int = 42,
    verbose: bool = True,
):
    """
    Treina o modelo época a época, monitorando treino e validação.
    Salva o checkpoint exclusivamente quando val_f1_macro atinge um novo máximo.
    Ao final, carrega os melhores pesos no modelo e retorna (model, history_df).
    """
    set_seed(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model, model_type=model_type, lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            images, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = accuracy_score(train_targets, train_preds)
        train_f1   = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Avaliação no conjunto de validação (seleção de modelo)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc  = val_metrics["acc"]
        val_f1   = val_metrics["f1_macro"]

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_f1_macro": train_f1,
            "val_f1_macro": val_f1,
        })

        if verbose and (epoch % 5 == 0 or epoch == 1 or is_best):
            star = " [Novo Melhor Val F1]" if is_best else ""
            print(
                f"Época [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.3f} || "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.3f} (Acc: {val_acc:.3f}){star}"
            )

    print(f"\nTreino finalizado. Melhor época no val: {best_epoch} (Val F1 Macro = {best_val_f1:.4f})")
    print(f"Pesos do melhor modelo salvos em: {save_path}")

    # Restaura os pesos da melhor época de validação
    model.load_state_dict(best_weights)
    history_df = pd.DataFrame(history)
    history_df.attrs["best_epoch"] = best_epoch
    history_df.attrs["best_val_f1"] = best_val_f1

    return model, history_df


def plot_learning_curves(history_df: pd.DataFrame, title_prefix: str = "Parte 0 (b) - ResNet-18", save_fig: str = None):
    """
    Parte 0 (item 3): Plota as curvas de Loss e F1 Macro (Treino vs. Validação),
    destacando a época do melhor Val F1 Macro e o ponto onde a rede decora o treino.
    """
    best_epoch = history_df.attrs.get("best_epoch", history_df["val_f1_macro"].idxmax() + 1)
    best_val_f1 = history_df["val_f1_macro"].max()

    # Identifica a primeira época em que o F1 de treino passa de 0.98 (decorou as 105 imagens)
    memorized_epochs = history_df[history_df["train_f1_macro"] >= 0.98]["epoch"]
    mem_epoch = memorized_epochs.iloc[0] if len(memorized_epochs) > 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Curva de Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Treino (105 imgs)", linewidth=2, color="#1f77b4")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validação (202 imgs)", linewidth=2, color="#d62728")
    axes[0].axvline(best_epoch, linestyle="--", color="green", alpha=0.8, label=f"Melhor Época Val ({best_epoch})")
    if mem_epoch is not None:
        axes[0].axvline(mem_epoch, linestyle=":", color="purple", alpha=0.8, label=f"Treino >= 98% F1 (Época {mem_epoch})")
    axes[0].set_title(f"{title_prefix} — Curva de Perda (Cross-Entropy)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Subplot 2: Curva de F1 Macro
    axes[1].plot(history_df["epoch"], history_df["train_f1_macro"], label="F1 Macro Treino", linewidth=2, color="#1f77b4")
    axes[1].plot(history_df["epoch"], history_df["val_f1_macro"], label="F1 Macro Validação", linewidth=2, color="#ff7f0e")
    axes[1].axvline(best_epoch, linestyle="--", color="green", alpha=0.8, label=f"Pico Val F1 = {best_val_f1:.3f} (Época {best_epoch})")
    if mem_epoch is not None:
        axes[1].axvline(mem_epoch, linestyle=":", color="purple", alpha=0.8, label=f"Decorou Treino (Época {mem_epoch})")
    axes[1].set_title(f"{title_prefix} — Curva de F1 Macro", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("F1 Macro")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title: str = "Matriz de Confusão", save_fig: str = None):
    """
    Parte 0 (item 2): Plota a Matriz de Confusão absoluta e normalizada por linha (recall por classe).
    """
    cm_abs  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(
        cm_abs, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False
    )
    axes[0].set_title(f"{title} (Contagem Absoluta)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Classe Prevista")
    axes[0].set_ylabel("Classe Verdadeira")
    axes[0].tick_params(axis='x', rotation=90)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names, ax=axes[1], vmin=0.0, vmax=1.0
    )
    axes[1].set_title(f"{title} (Normalizada por Linha / Recall)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Classe Prevista")
    axes[1].set_ylabel("Classe Verdadeira")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
    plt.show()


def log_budget_result(
    csv_path: str,
    metodo: str,
    orcamento: int,
    seed: int,
    f1_macro: float,
    acuracia: float
):
    """
    Salva ou atualiza uma linha no arquivo entregável curva_orcamento.csv
    Colunas exigidas no PA3: metodo, orcamento, seed, f1_macro, acuracia
    """
    new_row = pd.DataFrame([{
        "metodo": metodo,
        "orcamento": orcamento,
        "seed": seed,
        "f1_macro": round(float(f1_macro), 4),
        "acuracia": round(float(acuracia), 4),
    }])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Remove duplicata caso esteja re-rodando o mesmo experimento (metodo, orcamento, seed)
        mask = ~((df["metodo"] == metodo) & (df["orcamento"] == orcamento) & (df["seed"] == seed))
        df = pd.concat([df[mask], new_row], ignore_index=True)
    else:
        df = new_row

    df.sort_values(["metodo", "orcamento", "seed"], inplace=True)
    df.to_csv(csv_path, index=False)
    return df