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

ngpu = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using " + str(device))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            #64*8 x 4 x 4

            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            # 64*4 x 8 x 8

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            # 64*2 x 16 x 16

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #64 x 32 x 32

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        if str(device) is not 'cpu':
            x = nn.parallel.data_parallel(self.main, x, range(ngpu))
        else:
            x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            # 64 x 16 x 16
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2),

            # 128 x 8 x 8
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2),

            # 256 x 4 x 4
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2),

            # 512 x 2 x 2
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if str(device) is not 'cpu':
            x = nn.parallel.data_parallel(self.main, x, range(ngpu))
        else:
            x = self.main(x)
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
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

dataset = datasets.LSUN(root='./data/lsun', classes=['tower_train'], transform=trans)

discriminator = Discriminator()
x = torch.rand(2, 3, 64, 64)
#print("D_shape: ", discriminator.forward(x).shape)

generator = Generator()
x = torch.rand(2, 100, 1, 1)
#print("G_shape: ", generator.forward(x).shape)

dl = DataLoader(
    dataset,
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
                    './samples/lsun/%03d.png' % (epoch), normalize=True)
