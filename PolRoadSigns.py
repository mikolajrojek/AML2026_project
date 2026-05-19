import kagglehub
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

path = kagglehub.dataset_download("chriskjm/polish-traffic-signs-dataset")
path = path + r'\classification'
print("Path to dataset files:", path)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=[256, 256]), transforms.Normalize((0.5,),(0.5,))])

dataset = datasets.ImageFolder(root = path, transform=transforms)
train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2)

#decided to remove "other" signs as they would only introduce noise.
train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] != 20]
test_dataset = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i][1] != 20]
dataset.class_to_idx.pop('other')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
images, labels = next(iter(train_loader))

img_tensor, label = dataset[0]

print("Tensor shape:", img_tensor.shape) 
print("Dataset size:", len(train_dataset)+len(test_dataset))
print("Train dataset size: ", len(train_dataset))
print("Test dataset size: ", len(test_dataset)) 
print("Batch size:", train_loader.batch_size)
print("Number of batches per epoch:", len(train_loader))
print("Example label:", dataset[0][1])
print("Actual label:", dataset.classes[dataset[0][1]])  #it isn't found in the pytorch docs but ImageFolder automatically assignes
print("Example label:", dataset[80][1])                 #different folders to integers instead of their actual names as labels.
print("Actual label:", dataset.classes[dataset[80][1]])
print("Class mapping", dataset.class_to_idx)

fig, ax = plt.subplots(4, 4, figsize=(7,7))
ax = ax.flatten()

for i in range(16):
    img = images[i]
    img = img*0.5 + 0.5
    img = img.permute(1,2,0)
    img = img.numpy()
    ax[i].imshow(img)
    ax[i].set_title(dataset.classes[labels[i]])
    ax[i].axis('off')
plt.show()
plt.tight_layout()