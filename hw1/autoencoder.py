import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Hyper Parameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
ADAM_BETAS = (0.9, 0.999)
RELU_SLOPE = 0.1


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 128 x 128 x 32

            nn.Conv2d(32, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 64 x 64 x 32

            nn.Conv2d(32, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 32 x 32 x 64

            nn.Conv2d(64, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 16 x 16 x 64

            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 8 x 8 x 128

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 4 x 4 x 128

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.MaxPool2d(2),  # 2 x 2 x 128

            # nn.Conv2d(128, 256, 3, padding="same"),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(RELU_SLOPE, inplace=True),

            # nn.MaxPool2d(2),  # 1 x 1 x 256

            nn.Flatten(),  # 1 x 256
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            # nn.LeakyReLU(RELU_SLOPE),

            View((BATCH_SIZE, 128, 2, 2)),

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 4 x 4 x 128

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 8 x 8 x 128

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 32 x 32 x 64

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 64 x 64 x 32

            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 256 x 256 x 32

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 256 x 256 x 3

            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_ae():
    ae = AE()
    ae.apply(weights_init)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder("./dataset", transform=transform)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    loss_arr = []
    for i, data in enumerate(dataloader):
        input_images, _ = data
        optimizer.zero_grad()
        output_images = ae(input_images)
        loss = loss_fn(input_images, output_images)
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f'Iteration {i} has loss {loss}')
        if i % 100 == 0 and i != 0:
            plot_loss(np.arange(len(loss_arr)), loss_arr)
            imshow(torchvision.utils.make_grid(input_images), "input")
            imshow(torchvision.utils.make_grid(output_images), "output")


def imshow(img, title):
    # img = img / 2 + 0.5     # un-normalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(title)
    plt.show()


def plot_loss(x, y):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("|| D(E(x)) - x ||")
    plt.show()


if __name__ == '__main__':
    train_ae()
