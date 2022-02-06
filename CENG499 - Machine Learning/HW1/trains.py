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
test_set = CIFAR10(root="Dataset", train=False, transform=val_transform, download=True)

train_set_length = int(0.8 * len(train_set))
val_set_length = len(train_set) - train_set_length
train_set, val_set = random_split(train_set, [train_set_length, val_set_length])

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

"""
HYPERPARAMETERS...
I HAVE
    THREE DIFFERENT LEARNING RATES,
    THREE DIFFERENT FUNCTIONS,
    AND 1-LAYER, 2-LAYER, 3-LAYER NETWORKS, ALSO DIFFERENT FUNCTION PAIRS..

TO BE ABLE TO TRY ALL ACTIVATION FUNCTION
CONFIGURATIONS I SELECT 2 OF THEM FOR 3-LAYER
NETWORK AND CHANGE THEIR ORDERS
FOR EXAMPLE
case1 : relu -> tanh
case2 : tanh -> relu
case3 : hardswish -> relu
...
...
"""
learning_rates = [1e-3, 2e-4, 1e-4]
activate_functions = ["relu", "tanh", "hardwish"]
function_pairs = []
function_pairs.append([])
function_pairs.append(['relu'])
function_pairs.append(['tanh'])
function_pairs.append(['hardswish'])
function_pairs.append(['relu', 'relu'])
function_pairs.append(['relu', 'tanh'])
function_pairs.append(['relu', 'hardswish'])
function_pairs.append(['tanh', 'tanh'])
function_pairs.append(['tanh', 'relu'])
"""
AT LINE BELOW I REPEATED TANH -> TANH CONFIGURATION
NOT ON PURPOSE.. SO THAT THERE WILL BE 36 CONFIGURATION IN MY REPORT
"""
function_pairs.append(['tanh', 'tanh'])
function_pairs.append(['hardswish', 'hardswish'])
function_pairs.append(['hardswish', 'relu'])
function_pairs.append(['hardswish', 'tanh'])
configurations = []
for a in learning_rates:
    for f in function_pairs:
        configurations.append([a,f])
def print_configuration(conf):
    return "Learning Rate: " + str(conf[0]) + '\n' + "Functions: " + str(conf[1])
"""
NOW, TO GET CONFIGURATION SET,
WE DO CROSS-PRODUCT LEARNING RATES
AND ACTIVATION FUNCTIONS.
THIS YIELDS 13 x 3 = 39 CONFIGURATIONS
IN ALL MODELS I USED 4096 LENGTH FIRST-LAYER,
AND 2048 LENGTH SECOND-LAYER.
"""
print(configurations)

class MyModel(nn.Module):
    def __init__(self, number_of_ls, layer_lengths, forward_functions):
        super().__init__()
        print(number_of_ls, layer_lengths, forward_functions)
        self.lengths = layer_lengths
        self.forwards = forward_functions
        self.number_of_layers = number_of_ls
        if self.number_of_layers == 1:
            self.layer1 = nn.Linear(in_features=32*32, out_features=10)
        elif self.number_of_layers == 2:
            self.layer1 = nn.Linear(in_features=32*32, out_features=layer_lengths[0])
            self.layer2 = nn.Linear(in_features=layer_lengths[0], out_features=10)
        else:
            self.layer1 = nn.Linear(in_features=32*32, out_features=layer_lengths[0])
            self.layer2 = nn.Linear(in_features=layer_lengths[0], out_features=layer_lengths[1])
            self.layer3 = nn.Linear(in_features=layer_lengths[1], out_features=10)
 
    def forward(self, x):
        if self.number_of_layers == 1:
            x = torch.flatten(x, 1)
            x = self.layer1(x)
        elif self.number_of_layers == 2:
            x = torch.flatten(x, 1)
            if self.forwards[0] == "relu":
                x = F.relu(self.layer1(x))
            elif self.forwards[0] == "tanh":
                x = F.tanh(self.layer1(x))
            elif self.forwards[0] == "hardswish":
                x = F.hardswish(self.layer1(x))
            x = self.layer2(x)
        else:
            x = torch.flatten(x, 1)
            if self.forwards[0] == "relu":
                x = F.relu(self.layer1(x))
            elif self.forwards[0] == "tanh":
                x = F.tanh(self.layer1(x))
            elif self.forwards[0] == "hardswish":
                x = F.hardswish(self.layer1(x))
    
            if self.forwards[1] == "relu":
                x = F.relu(self.layer2(x))
            elif self.forwards[1] == "tanh":
                x = F.tanh(self.layer2(x))
            elif self.forwards[1] == "hardswish":
                x = F.hardswish(self.layer2(x))
            x = self.layer3(x)
        return x

# logits --> unnormalized probabilites
nn.BatchNorm1d
nn.Dropout


device = 'cuda' if torch.cuda.is_available() else 'cpu'

acs = open("accuracies.txt","w")
model_amount = len(configurations)
loss_function = nn.CrossEntropyLoss()
"""
THIS IS MAX NUMBER OF EPOCHS..
I DO EARLY STOP BY HOLDING AVERAGES
OF VALIDATION LOSS..
"""
num_epochs = 30
for mdl in range(len(configurations)):
    model = MyModel(len(configurations[mdl][1])+1,[4096,2048],configurations[mdl][1]).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=configurations[mdl][0])
    acs.write("Model: " + str(mdl) + ' ' + str(print_configuration(configurations[mdl]) + '\n'))
    vals = []
    ep = 0
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        accum_train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)
            # accumlate the loss
            accum_train_loss += loss.item()
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        accum_val_loss = 0
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                accum_val_loss += loss_function(output, labels).item()
        
        acs.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}'+'\n')
        """
        HERE I SAVE VALIDATION LOSSES
        AFTER EVERY EPOCH..
        IF NEW RESULT IS WORSE THAN AVERAGE OF THEM
        I DO 'EARLY STOP'
        """        
        if ep>5 and accum_val_loss > sum(vals)/ep:
            break
        ep += 1
        vals.append(accum_val_loss)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)
    acs.write(f'Val Accuracy = {100 * correct/total :.3f}%\n')
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)
    acs.write(f'Test Accuracy = {100 * correct/total :.3f}%\n\n')

acs.close()