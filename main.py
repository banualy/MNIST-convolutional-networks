import torch
import torch.nn as nn #neural network models, loss and activation functions
import torch.optim as optim #optimization algorithms
import torch.nn.functional as F #activation functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Creating convolutional network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__() #initialization of method
        self.con1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        #n_out=(n_in+2p-k)/s+1
        self.pool =  nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.con2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1),
                              stride=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = self.pool(x)
        x = F.relu(self.con2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channels = 1
num_classes=10
learning_rate= 0.001
batch_size=64
num_epochs=6

#loading data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#initializing network
model = CNN().to(device)

#loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # getting data to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set gradients to zero in each batch
        loss.backward()

        # gradient descent step
        optimizer.step()


# checking accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Training data")
    else:
        print("Test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions =scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(num_correct / num_samples, f'{float(num_correct) / float(num_samples) * 100:.2f} accuracy')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

