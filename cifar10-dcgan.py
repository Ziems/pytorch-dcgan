import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using " + str(device))

# Based on
# https://medium.com/@stepanulyanin/dcgan-adventures-with-cifar10-905fb0a24d21
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * 8)
        self.act1 = nn.ReLU()
        #64*8 x 4 x 4

        self.conv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 4)
        self.act3 = nn.ReLU()
        # 64*4 x 8 x 8

        self.conv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.act4 = nn.ReLU()
        # 64*2 x 16 x 16

        self.conv4 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        #64 x 32 x 32

        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.act6 = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act3(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act5(x)

        x = self.conv5(x)
        x = self.act6(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.act1 = nn.LeakyReLU(0.2)

        # 64 x 16 x 16
        self.conv2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.act2 = nn.LeakyReLU(0.2)

        # 128 x 8 x 8
        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.act3 = nn.LeakyReLU(0.2)

        # 256 x 4 x 4
        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*8)
        self.act4 = nn.LeakyReLU(0.2)

        # 512 x 2 x 2
        self.conv5 = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.act5 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act4(x)
        
        x = self.conv5(x)
        x = self.act5(x)
        return x.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

trans = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

mnist_trainset = datasets.CIFAR10(root='./data/cifar10', download=True, transform=trans)

discriminator = Discriminator()
x = torch.rand(2, 3, 64, 64)
print("D_shape: ", discriminator.forward(x).shape)

generator = Generator()
x = torch.rand(2, 100, 1, 1)
print("G_shape: ", generator.forward(x).shape)

dl = DataLoader(
    mnist_trainset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

loss = nn.BCELoss()

d_optim = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

g_optim = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

discriminator = discriminator.to(device)
discriminator.apply(weights_init)

generator = generator.to(device)
generator.apply(weights_init)

N_EPOCHS = 25

for epoch in range(N_EPOCHS):
    D_losses = []
    G_losses = []
    for i, data in enumerate(dl, 0):
        # Train discriminator
        discriminator.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1, device=device)

        output = discriminator(real_cpu)
        errD_real = loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(0)
        output = discriminator(fake.detach())
        errD_fake = loss(output, label)
        errD_fake.backward()
        d_optim.step()
        D_losses.append((errD_fake.data.item() + errD_real.data.item()) / 2)

        # Train generator
        generator.zero_grad()
        label.fill_(1)
        output = discriminator(fake)
        errG = loss(output, label)
        errG.backward()
        g_optim.step()
        G_losses.append(errG.data.item())

        if i % 100 == 0:
            print("Epoch: ", epoch+1)
            print("D_loss: ", D_losses[-1])
            print("G_loss: ", G_losses[-1])
            fake = generator(torch.randn(64, 100, 1, 1, device=device))
            vutils.save_image(fake.detach(),
                    './samples/%03d.png' % (epoch), normalize=True)
