from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
import torch.optim as optim
from EMNIST.emnist_dataset import EMnistDataset
import matplotlib.pyplot as plt

# train_data = datasets.EMNIST('C:/Users/mosto/PycharmProjects/data', train=True, download=False, split='mnist')
train_data = EMnistDataset(10000, 42)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 Hidden Layer Network
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc5 = nn.Linear(512, 10)

        # Dropout module with 0.2 probbability

        # Add softmax on output layer
        self.log_softmax = F.softmax

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = self.log_softmax(self.fc5(x))

        return x


batch_size = 100

train, valid = train_test_split(train_data, test_size=0.2, random_state=42)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

DEVICE = 'cpu'
# Instantiate our model
model = Classifier()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)
epochs = 100
steps = 0
print_every = 50
train_losses, test_losses = [], []
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in train_loader:
#         # images = images.view(100, 784)
#         # images = images.to(DEVICE, dtype=torch.float)
#         # labels = labels.to(DEVICE, dtype=torch.float)
#         steps += 1
#         # Prevent accumulation of gradients
#         optimizer.zero_grad()
#
#         # Make predictions
#         log_ps = model(images)
#
#         loss = criterion(log_ps, labels)
#         running_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     print(running_loss)
# torch.save(model, 'model.pth')

model = torch.load('model.pth')
test_loss = 0.0
accuracy = 0.0

for images, labels in valid_loader:
    images = images.to(DEVICE, dtype=torch.float)
    labels = labels.to(DEVICE, dtype=torch.float)
    log_ps = model(images)
    test_loss += criterion(log_ps, labels)

    ps = log_ps

    top_p, top_class = ps.topk(1, dim=1)
    l = [(i == 1).nonzero(as_tuple=True)[0] for i in labels]
    equals = (top_class.flatten()) == (torch.tensor(l))
    equals = list(map(lambda x: 1 if x == True else 0, equals))
    print('Batch accuracy:', equals.count(1)/100)

#Модель на торче - 0.7-0.8