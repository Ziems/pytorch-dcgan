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

# Based on
# https://medium.com/@stepanulyanin/dcgan-adventures-with-cifar10-905fb0a24d21
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 2 * 2 * 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.activation1 = nn.LeakyReLU()
        # 256 x 2 x 2

        self.conv1 = nn.ConvTranspose2d(256, 128, (5, 5), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activation2 = nn.LeakyReLU()
        # 128 x 4 x 4

        self.conv2 = nn.ConvTranspose2d(128, 64, (5, 5))
        self.bn3 = nn.BatchNorm2d(64)
        self.activation3 = nn.LeakyReLU()
        # 64 x 8 x 8

        self.conv3 = nn.ConvTranspose2d(64, 32, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.activation4 = nn.LeakyReLU()
        # 32 x 16 x 16

        self.logits = nn.ConvTranspose2d(32, 3, (5, 5), stride=2, padding=2, output_padding=1)
        # 3 x 32 x 32
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = x.view(-1, 256, 2, 2)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = self.conv3(x)
        x = self.bn4(x)
        x = self.activation4(x)

        x = self.logits(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, (5, 5), 2, padding=2)
        self.act1 = nn.LeakyReLU()

        # 32 x 16 x 16
        self.conv2 = nn.Conv2d(32, 64, (5, 5), 2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()

        # 64 x 8 x 8
        self.conv3 = nn.Conv2d(64, 128, (5, 5), 2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()

        # 128 x 4 x 4
        self.conv4 = nn.Conv2d(128, 256, (5, 5), 2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act4 = nn.LeakyReLU()

        # 256 x 2 x 2
        self.fc1 = nn.Linear(256 * 2 * 2, 1)
        self.act5 = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.act1(x)
        print("X Shape0: ", x.shape)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)
        print("X Shape1: ", x.shape)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.act3(x)
        print("X Shape2: ", x.shape)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act4(x)
        print("X Shape3: ", x.shape)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc1(x)
        x = self.act5(x)
        return x

trans = transforms.Compose([
    transforms.ToTensor()
])

mnist_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=trans)
mnist_testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=None)

discriminator = Discriminator()
x = torch.rand(2, 3, 32, 32)
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
    generator.eval()
    sample_generator(generator, epoch)
    discriminator.train()
    generator.train()

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
