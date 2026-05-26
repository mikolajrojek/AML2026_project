import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

from data import data_trafic_signs
from model import SimpleCNN, BetterCNN, SimpleButBetterCNN
from training import train,validate


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader, test_loader = data_trafic_signs(16,16)    

    model = BetterCNN().to(device)
    
    train(model=model, lr=0.001, train_loader=train_loader, bs=16, epochs = 10)

    validate(model, test_loader)


main()

