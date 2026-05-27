import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

from data import data_trafic_signs
from model import SimpleCNN, BetterCNN
from training import train,validate
from hyperparameter_optimisation import hyp_optimisation


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # train_loader, test_loader = data_trafic_signs(16,16)    

    model = BetterCNN().to(device)

    lr_tab = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    bs_tab = [4, 16, 32, 64, 128]

    # lr_tab = [0.01, 0.005, 0.001]
    # bs_tab = [16, 32]

    hyp_optimisation(lr_tab, bs_tab)


    
    # train(model=model, lr=0.001, train_loader=train_loader, bs=16, epochs = 10)

    # validate(model, test_loader)


main()

