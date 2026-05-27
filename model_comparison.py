import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

from training import train, validate
from model import ModelCNN_ReLU, ModelCNN_SiLU, ModelCNN_LeakyReLU, ModelCNN_GELU, ModelCNN_ELU

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10

def comparison(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [ModelCNN_ReLU, ModelCNN_SiLU, ModelCNN_LeakyReLU, ModelCNN_GELU, ModelCNN_ELU]

    fl_tab = []
    acc_tab = []

    for m in models:
        torch.manual_seed(42)
        model = m().to(device)

        final_loss = train(model, LEARNING_RATE, train_loader, BATCH_SIZE, EPOCHS)
        fl_tab.append(final_loss)
        accuracy = validate(model, test_loader)
        acc_tab.append(accuracy)
    i = 1
    print("Model 1: ReLU, 2: SiLu, 3: LeakyReLU, 4: GELU, 5: ELU")
    for acc,fl in zip(acc_tab,fl_tab):
        print(f"Model {i}: accuracy {acc}, final loss: {fl}")
        i += 1












