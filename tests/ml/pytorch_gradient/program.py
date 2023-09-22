import time
from tqdm import tqdm
import numpy as np
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 25)
        self.fc2 = nn.Linear(25, 1)

        self.ordered_layers = [self.fc1, self.fc2]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        outputs = self.fc2(x)
        return outputs


def train_model(model, criterion, optimizer, num_epochs, with_clip=True):
    since = time.time()
    dataset_size = 1000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        time.sleep(0.1)
        model.train()  # Set model to training mode

        running_loss = 0.0
        batch_norm = []

        # Iterate over data.
        for idx, (inputs, label) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)

            # zero the parameter gradients
            # no issues should be found here!
            optimizer.zero_grad()

            # forward
            logits = model(inputs)
            loss = criterion(logits, label)

            # backward
            loss.backward()

            # Gradient Value Clipping, will prevent issues
            if with_clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            # calculate gradient norms
            for layer in model.ordered_layers:
                norm_grad = layer.weight.grad.norm()
                batch_norm.append(norm_grad.numpy())

            optimizer.step()  # DyLin warn
            # hook has to be after optimizer.step() !

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_size

        print('Train Loss: {:.4f}'.format(epoch_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


device = torch.device("cpu")

# prepare data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=1)

X = torch.Tensor(X)
y = torch.Tensor(y)

dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

model = Net().to(device)
model.apply(init_weights)
optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.8)
criterion = nn.MSELoss()

norms = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=5,
    with_clip=False,  # make it True to use clipping and remove issues
)
