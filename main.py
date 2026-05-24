import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))

from data import data_trafic_signs
from model import SimpleCNN


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader, test_loader = data_trafic_signs(16,16)

    model = SimpleCNN().to(device)


main()

