import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ENA24UnsupervisedDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Caminho para a pasta principal.
            split (str): 'train', 'val' ou 'test_public'.
            transform: Transformacoes do torchvision para a imagem.
        """
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.transform = transform
        
        # Filtra para garantir que so vamos ler arquivos de imagem
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = [
            f for f in os.listdir(self.image_dir) 
            if f.lower().endswith(valid_extensions)
        ]

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
        
        return image, '.'.join(img_name.split(".")[:-1])
    
    
    
def get_labels(
    root: str, 
    name_file: str=None, 
    labels_qtt: int=5,
) -> pd.DataFrame:
    """ Faz a leitura e carregamento dos labels em duas opcoes:
    - Via numero de labels, quando o arquivo tem o nome 'rotulo_{1|5|20}'
    - Via nome do arquivo, especialmente para os casos 'test_public|val'

    Args:
        root (str): Path onde se encontram as imagens e os labels.
        name_file (str, optional): Nome do arquivo de labels. Defaults to None.
        labels_qtt (int, optional): Quantidade de labels de treino. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame de labels.
    """
    if not (name_file is None):
        return pd.read_csv(f"{root}/labels/{name_file}.csv")
    return pd.read_csv(f"{root}/labels/rotulos_{labels_qtt}.csv")