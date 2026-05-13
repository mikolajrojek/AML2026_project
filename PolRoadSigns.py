import kagglehub
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

path = kagglehub.dataset_download("chriskjm/polish-traffic-signs-dataset")
path = path + r'\classification' #r necessary cuz \c is a command like \n and python reads it as such, else causes a warning.
print("Path to dataset files:", path)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=[96, 96])]) #probably will add more stuff there later. 96x96 is temporary
# this resize is necessary because we need same shape tensors to put into a neural network. math.

dataset = datasets.ImageFolder(root = path, transform=transforms) #ImageFolder has no train or download methods.
loader = DataLoader(dataset, batch_size=16, shuffle=True) #train-test split done later cuz im lazy. bs=16 is temporary.
images, labels = next(iter(loader))

img_tensor, label = dataset[0]

print("Tensor shape:", img_tensor.shape) 
print("Dataset size:", len(dataset)) 
print("Batch size:", loader.batch_size)
print("Number of batches per epoch:", len(loader))
print("Dataset size:", len(dataset))
print("Example label:", dataset[0][1])
print("Actual label:", dataset.classes[dataset[0][1]])  #it isn't found in the pytorch docs but ImageFolder automatically assignes
print("Example label:", dataset[80][1])                 #different folders to integers instead of their actual names as labels.
print("Actual label:", dataset.classes[dataset[80][1]])
print("Class mapping", dataset.class_to_idx)

img_raw, label = dataset[82]

print("Label:", label, dataset.classes[dataset[80][1]])
plt.imshow(img_raw.permute(1,2,0))
plt.title(f"Raw image - label: {label, dataset.classes[dataset[82][1]]}")
plt.axis("off")
plt.show()