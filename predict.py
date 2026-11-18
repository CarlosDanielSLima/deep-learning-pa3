import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models import build_resnet18, count_parameters, LinearProbe, GradCAMLinearProbeWrapper
from src.data_loader import get_transforms, get_class_mappings
import time

def load_best_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Carrega o modelo do checkpoint especificado.
    Usa pretrained=False para garantir que roda SEM internet (não tenta baixar do ImageNet).
    """
    backbone = build_resnet18(pretrained=False)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    
    model = LinearProbe(backbone=backbone, num_classes=21)
    
    # Carrega os pesos do checkpoint treinado
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    final_model = GradCAMLinearProbeWrapper(model).to(device)
    return final_model


class InferenceDataset(Dataset):
    """Dataset simplificado apenas para varrer uma pasta de imagens na inferência"""
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        image_id = os.path.splitext(img_name)[0]
        return image, image_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagens", type=str, required=True, help="PASTA de onde carregar as imagens")
    parser.add_argument("--saida", type=str, required=True, help="Arquivo CSV de saída")
    args = parser.parse_args()

    # Marca o início do tempo de execução
    start_time = time.time()

    # Define dispositivo e caminhos fixos
    device = torch.device("cpu")
    root_dir = "pa3_dados"
    checkpoint_path = "checkpoints/swav_imgnet_experiment_random_crap.pt"

    print(f"Rodando em: {device} (Offline)")

    # Carrega classes para pegar os nomes em texto
    _, idx_to_label, class_names = get_class_mappings(root_dir)
    print(f"Total de classes carregadas: {len(class_names)}")

    # Carrega modelo e imprime parâmetros
    model = load_best_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    
    n_params = count_parameters(model)
    print(f"Número de parâmetros do modelo: {n_params:,}")
    
    if n_params > 12000000:
        print("AVISO: O modelo excedeu o limite de 12 milhões de parâmetros!")

    # Pipeline de dados apontando para a PASTA do terminal
    _, eval_transform = get_transforms(img_size=(256, 352))
    dataset = InferenceDataset(args.imagens, transform=eval_transform)
    
    # Batch size 16 (bom equilíbrio) para CPU não travar
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Inferência
    results = []
    print(f"Iniciando inferência na pasta: {args.imagens}...")
    
    with torch.no_grad():
        for images, image_ids in tqdm(loader, desc="Progresso da Inferência"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for img_id, pred in zip(image_ids, preds):
                results.append({
                    "image_id": img_id,
                    "label": idx_to_label[pred.item()]
                })

    # Exporta para CSV
    df = pd.DataFrame(results)
    df.to_csv(args.saida, index=False)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"Pronto! Arquivo gerado: {args.saida}")
    print(f"Tempo total de execução: {elapsed_time:.2f} minutos.")

if __name__ == "__main__":
    main()