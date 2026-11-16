import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Estatísticas oficiais do ImageNet para normalização de modelos pré-treinados
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    """Fixa todas as seeds para garantir reprodutibilidade nos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_mappings(root: str):
    """
    Lê classes.csv e retorna:
    - label_to_idx (dict): nome_da_classe (str) -> idx (int)
    - idx_to_label (dict): idx (int) -> nome_da_classe (str)
    - class_names (list): lista ordenada de nomes das 21 classes
    """
    df_classes = pd.read_csv(os.path.join(root, "classes.csv")).sort_values("idx")
    label_to_idx = dict(zip(df_classes["label"], df_classes["idx"]))
    idx_to_label = dict(zip(df_classes["idx"], df_classes["label"]))
    class_names = df_classes["label"].tolist()
    return label_to_idx, idx_to_label, class_names


def get_labels(
    root: str, 
    file_name: str = None, 
    labels_qtt: int = 5,
) -> pd.DataFrame:
    """
    Faz a leitura e carregamento dos labels em duas opções:
    - Via número de labels, quando o arquivo tem o nome 'rotulos_{1|5|20}.csv'
    - Via nome do arquivo, especialmente para os casos 'test_public' ou 'val'
    """
    if file_name is not None:
        return pd.read_csv(os.path.join(root, "labels", f"{file_name}.csv"))
    return pd.read_csv(os.path.join(root, "labels", f"rotulos_{labels_qtt}.csv"))

# Motivos para usar (256, 352) como tamanho de imagem: Mantém a proporção original das imagens (4:3) sem distorção e garante que nenhuma camada convolucional arredonde ou perca bordas na divisão inteira ao dividir por 2 5vezes.
def get_transforms(img_size: tuple = (256, 352)):
    """
    Retorna os pipelines de transformação (train_transform, eval_transform).
    - Mantém 3 canais RGB e normaliza com média/desvio do ImageNet.
    - No treino, usa aumentações que preservam o animal no quadro (sem cortes agressivos).
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),  # Ajuda a generalizar entre fotos diurnas e IR noturnas
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


class ENA24UnsupervisedDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.transform = transform
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.lower().endswith(valid_extensions)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Converte para RGB para padronizar canais
        image = Image.open(img_path).convert("RGB")

        # Aplica as transformacoes
        if self.transform:
            image = self.transform(image)

        image_id = os.path.splitext(img_name)[0]
        return image, image_id


class ENA24SupervisedDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        file_name: str = None, 
        labels_qtt: int = 5, 
        split: str = 'train', 
        transform=None,
        return_id: bool = False
    ):
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.transform = transform
        self.return_id = return_id
        
        # Mapeamento oficial de classes.csv (str -> int)
        self.label_to_idx, self.idx_to_label, self.class_names = get_class_mappings(root_dir)
        
        # Coleta os labels do CSV correspondente
        df_labels = get_labels(root=root_dir, file_name=file_name, labels_qtt=labels_qtt)
        self.image_ids   = df_labels["image_id"].astype(str).tolist()
        self.image_files = [f"{img_id}.jpg" for img_id in self.image_ids]
        self.labels      = [self.label_to_idx[lbl] for lbl in df_labels["label"]]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.return_id:
            return image, label, self.image_ids[idx]
        return image, label


def get_supervised_dataloaders(
    root_dir: str,
    labels_qtt: int = 5,
    batch_size: int = 16,
    img_size: tuple = (256, 352),
    num_workers: int = 2,
    seed: int = 42
):
    """
    Cria e retorna os DataLoaders supervisionados de treino, validação e teste público.
    Garante shuffle=True apenas no treino e shuffle=False em val e test_public.
    """
    set_seed(seed)
    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_ds = ENA24SupervisedDataset(
        root_dir=root_dir, file_name=None, labels_qtt=labels_qtt, split='train', transform=train_tf
    )
    val_ds = ENA24SupervisedDataset(
        root_dir=root_dir, file_name='val', split='val', transform=eval_tf
    )
    test_ds = ENA24SupervisedDataset(
        root_dir=root_dir, file_name='test_public', split='test_public', transform=eval_tf
    )

    g = torch.Generator()
    g.manual_seed(seed)

    # Em Windows com Jupyter, se num_workers > 0 travar, basta passar num_workers=0
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, test_loader