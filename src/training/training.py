import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from PIL import Image
# if not hasattr(Image, 'Resampling'):
#     Image.Resampling = Image




def train(model , lr,  train_loader, bs, epochs = 10, seed = None, sett = None, md = None):

    # Create a TensorBoard writer
    if seed == None and md == None:
        writer = SummaryWriter(log_dir=f"runs/lr_{lr}_bs_{bs}")
    elif md != None:
        writer = SummaryWriter(log_dir=f"runs/lr_{lr}_bs_{bs}_model_{md}")
    else:
        writer = SummaryWriter(log_dir=f"runs/lr_{lr}_bs_{bs}_seed_{seed}_Sett_{sett}")


    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.flush()


        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")






    # Example: log images (first batch of training data)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    img_grid = vutils.make_grid(images[:8])
    writer.add_image("Sample Inputs", img_grid, 0)


    # Example: log weight histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, epoch)




    # log hyperparameters
    hparams = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": train_loader.batch_size,
        "optimizer": "Adam",
        "epochs": epochs
    }


    metrics = {
        "hparam/train_loss": epoch_loss
    }


    writer.add_hparams(hparams, metrics)
    return running_loss / len(train_loader)


    # Run `tensorboard --logdir=runs` to visualize (in terminal)


def validate(model, test_loader):
    model.eval()
    correct = 0
    total = 0


    with torch.no_grad(): # disables gradient tracking
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()


    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    


    # import matplotlib.pyplot as plt


    # # Switch model to evaluation mode
    # model.eval()


    # # Pick a batch from test set
    # data_iter = iter(test_loader)
    # images, labels = next(data_iter)
    # images, labels = images.to(device), labels.to(device)


    # # Forward pass
    # outputs = model(images)
    # _, preds = torch.max(outputs, 1)


    # # Display first 8 images with predictions
    # fig, axes = plt.subplots(2, 4, figsize=(12,6))
    # for i, ax in enumerate(axes.flat):
    #     img = images[i].cpu().squeeze()
    #     ax.imshow(img, cmap='gray')
    #     ax.set_title(f"Pred: {preds[i].item()} / True: {labels[i].item()}")
    #     ax.axis('off')
    # plt.show()


    # # Example: Inference on single image
    # single_img = images[0].unsqueeze(0)  # add batch dimension
    # output_single = model(single_img)
    # pred_single = torch.argmax(output_single, dim=1)
    # print(f"Single image prediction: {pred_single.item()}, True label: {labels[0].item()}")

    return 100 * correct / total