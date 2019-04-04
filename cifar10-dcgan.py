import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using " + str(device))

def sample_generator(model, index):
    z = torch.randn((2, 100)).view(-1, 100, 1, 1).to(device)
    image = model(z)[0].cpu()
    image = transforms.ToPILImage(mode='RGB')(image)
    if not os.path.exists('./samples/'):
        os.makedirs('./samples/')
    image.save("./samples/" + str(index) + ".png")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 1024)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(1024, 256*8*8)
        self.bn1 = nn.BatchNorm1d(num_features=256*8*8)
        self.activation2 = nn.Tanh()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(5, 5), padding=0)
        self.activation3 = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=(5, 5), padding=2)
        self.activation4 = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.bn1(x)
        x = self.activation2(x)
        x = x.view(-1, 256, 8, 8)
        x = F.interpolate(x, (16, 16))
        x = self.conv1(x)
        x = self.activation3(x)
        x = F.interpolate(x, (32, 32))
        x = self.conv2(x)
        x = self.activation4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.activation1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.activation2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(128*6*6, 1024)
        self.activation3 = nn.Tanh()
        self.fc2 = nn.Linear(1024, 512)
        self.activation4 = nn.Tanh()
        self.fc3 = nn.Linear(512, 1)
        self.activation5 = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 128 * 6 * 6) # flatten
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)
        x = self.activation4(x)
        x = self.fc3(x)
        x = self.activation5(x)
        return x

trans = transforms.Compose([
    transforms.ToTensor()
])

mnist_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=trans)
mnist_testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=None)

discriminator = Discriminator()
x = mnist_trainset[13][0]
discriminator.forward(x)

generator = Generator()
x = torch.rand(2, 100)
generator.forward(x)

train_dl = DataLoader(
    mnist_trainset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

test_dl = DataLoader(
    mnist_testset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)

loss = nn.BCELoss()

d_optim = torch.optim.SGD(
    discriminator.parameters(),
    lr=0.0005,
    momentum=0.9,
    nesterov=True
)

g_optim = torch.optim.SGD(
    generator.parameters(),
    lr=0.0005,
    momentum=0.9,
    nesterov=True
)

discriminator = discriminator.to(device)
generator = generator.to(device)

N_EPOCHS = 20

for epoch in range(N_EPOCHS):
    discriminator.train()
    generator.train()
    sample_generator(generator, epoch)

    D_losses = []
    G_losses = []

    for X, _ in train_dl:
        X = X.to(device)

        d_optim.zero_grad()
        mini_batch = X.size()[0]

        y_real = torch.ones(mini_batch).to(device)
        y_fake = torch.zeros(mini_batch).to(device)

        discriminator_res = discriminator(X).squeeze()
        discriminator_real_loss = loss(discriminator_res, y_real)

        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
        generator_res = generator(z)

        discriminator_res = discriminator(generator_res).squeeze()
        discriminator_fake_loss = loss(discriminator_res, y_fake)

        discriminator_fake_score = discriminator_res.data.mean()

        discriminator_train_loss = discriminator_real_loss + discriminator_fake_loss

        discriminator_train_loss.backward()
        d_optim.step()

        D_losses.append(discriminator_train_loss.data.item())
        print("D_loss: ", D_losses[-1])

        g_optim.zero_grad()
        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)

        G_res = generator(z)
        D_res = discriminator(G_res).squeeze()
        G_train_loss = loss(D_res, y_real)
        G_train_loss.backward()
        g_optim.step()

        G_losses.append(G_train_loss.data.item())
        print("G_loss: ", G_losses[-1])

    print("Epoch: ", epoch + 1)
