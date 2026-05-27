import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

from data import data_trafic_signs
from model import ModelCNN_ReLU
from training import train,validate
from hyperparameter_optimisation import hyp_optimisation
from model_comparison import comparison

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_dataset, test_dataset = data_trafic_signs()
    test_loader = DataLoader(test_dataset, batch_size=16)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    comparison(train_loader, test_loader)
    # model = ModelCNN_ReLU().to(device)
    
    # train(model=model, lr = LEARNING_RATE,bs=BATCH_SIZE, train_loader=train_loader, epochs = EPOCHS)

    # validate(model, test_loader)


main()

