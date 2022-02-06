# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm


# Fix the randomness
seed = 1234
torch.manual_seed(seed)

train_transform = T.Compose([
    T.ToTensor(),
    T.Grayscale(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])

val_transform = test_transform = T.Compose([
    T.ToTensor(),
    T.Grayscale(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])

train_set = CIFAR10(root="Dataset", train=True, transform=train_transform, download=True)
train_set_length = int(0.5 * len(train_set))
val_set_length = len(train_set) - train_set_length
train_set, val_set = random_split(train_set, [train_set_length, val_set_length])
test_set = CIFAR10(root="Dataset", train=False, transform=val_transform, download=True)

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)



# Define the ANN
class MyModel(nn.Module):
    def __init__(self, num_layer1, num_layer2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.hardswish(self.layer1(x))
      x = F.hardswish(self.layer2(x))
      x = self.layer3(x)
      # not have to add softmax layer here
      return x

# logits --> unnormalized probabilites
nn.BatchNorm1d
nn.Dropout



# Instantiate the model and Train it for 3 epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyModel(200, 200).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 3
for epoch in tqdm(range(num_epochs)):
    ##### NO TRAINING
    # Validation
    model.eval()
    accum_val_loss = 0
    with torch.no_grad():
        for j, (imgs, labels) in enumerate(val_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            accum_val_loss += loss_function(output, labels).item()

    # print statistics of the epoch
    print(f'Epoch = {epoch} | Train Loss = {0:.4f}\tVal Loss = {accum_val_loss / j:.4f}')


# Compute Test Accuracy
model.eval()
with torch.no_grad():
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        
        _, predicted_labels = torch.max(output, 1)
        correct += (predicted_labels == labels).sum()
        total += labels.size(0)

print(f'Test Accuracy = {100 * correct/total :.3f}%')