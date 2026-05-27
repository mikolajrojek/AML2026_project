import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

from data import data_trafic_signs
from training import train, validate
from model import BetterCNN


def draw_heatmap(data_lr, data_bs, data_fl, data_acc):
    """Fixed heatmap function for accuracy/loss vs LR & batch size"""
   
    # Create proper meshgrids (shape: lr x bs)
    lr_grid, bs_grid = np.meshgrid(data_lr, data_bs)
   
    # Build grids matching meshgrid shape
    acc_grid = np.array([[data_acc[(bs, lr)] for lr in data_lr] for bs in data_bs])
    fl_grid = np.array([[data_fl[(bs, lr)] for lr in data_lr] for bs in data_bs])
   
    # 1-row 2-col subplots (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Accuracy & Loss vs Batch Size & Learning Rate', fontsize=14, fontweight='bold')
   
    # Plot 1: Accuracy
    c0 = axes[0].contourf(lr_grid, bs_grid, acc_grid, levels=20, cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='Test Accuracy', shrink=0.8)
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Batch Size')
    axes[0].set_xscale('log')
    axes[0].set_title('Test Accuracy')
    axes[0].grid(True, alpha=0.3)
   
    # Plot 2: Final Loss
    c1 = axes[1].contourf(lr_grid, bs_grid, fl_grid, levels=20, cmap='plasma_r')  # Reversed for loss
    fig.colorbar(c1, ax=axes[1], label='Final Loss', shrink=0.8)
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Batch Size')
    axes[1].set_xscale('log')
    axes[1].set_title('Final Loss')
    axes[1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.savefig('hyperparam_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()




def hyp_optimisation(lr_tab, bs_tab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = data_trafic_signs()
    test_loader = DataLoader(test_dataset, batch_size=16)

    accuracy_tab = {}
    final_loss_tab = {}

    for lr in lr_tab:
        for bs in bs_tab:
            train_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
            torch.manual_seed(42)
            model = BetterCNN().to(device)

            final_loss = train(model, lr, train_loader, bs = bs, epochs=10)
            accuracy = validate(model, test_loader)

            final_loss_tab[(bs,lr)] = final_loss
            accuracy_tab[(bs,lr)] = accuracy

    draw_heatmap(lr_tab, bs_tab, final_loss_tab, accuracy_tab)



    


