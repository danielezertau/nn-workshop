import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATASET_DIR = "./dataset"
RESULTS_DIR = "results"

# Hyper Parameters
NUM_TEST_IMAGES = 1000
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.999)
RELU_SLOPE = 0.1
NUM_EPOCHS = 10


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.MaxPool2d(2),  # 128 x 128 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(32, 32, 3, padding="same"),
            nn.MaxPool2d(2),  # 64 x 64 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(32, 64, 3, padding="same"),
            nn.MaxPool2d(2),  # 32 x 32 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(64, 64, 3, padding="same"),
            nn.MaxPool2d(2),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(64, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 4 x 4 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 2 x 2 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Conv2d(128, 256, 3, padding="same"),
            nn.MaxPool2d(2),  # 1 x 1 x 256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),

            nn.Flatten(),  # 1 x 256
        )

        self.decoder = nn.Sequential(
            View((-1, 256, 1, 1)),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),  # 16 x 16 x 64

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


def train_test_ae():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae = AE()
    ae.to(device)
    ae.apply(weights_init)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    test_dataset = Subset(dataset, np.arange(NUM_TEST_IMAGES))
    train_dataset = Subset(dataset, np.arange(NUM_TEST_IMAGES, len(dataset)))
    test_dataloader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False)
    train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)

    train_loss_arr = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")
        for i, data in enumerate(train_dataloader, start=1):
            input_images = data[0].to(device)
            optimizer.zero_grad()
            output_images = ae(input_images)
            loss = loss_fn(input_images, output_images)
            train_loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f'Training epoch {epoch} iteration {i} has loss {loss}')
            if i % 500 == 0:
                plot_loss(np.arange(len(train_loss_arr)), train_loss_arr, f"train_loss_{epoch}_{i}.pdf")
                imshow(torchvision.utils.make_grid(input_images), "input", f"train_input_{epoch}_{i}.pdf")
                imshow(torchvision.utils.make_grid(output_images), "output", f"train_output_{epoch}_{i}.pdf")

        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, start=1):
                input_images = data[0].to(device)
                output_images = ae(input_images)
                loss = loss_fn(input_images, output_images)
                test_loss += loss.item()
                if i % 10 == 0:
                    imshow(torchvision.utils.make_grid(input_images), "input", f"test_input_{epoch}_{i}.pdf")
                    imshow(torchvision.utils.make_grid(output_images), "output", f"test_output_{epoch}_{i}.pdf")
            print(f'Epoch {epoch} average test loss: {test_loss / len(test_dataloader)}')


def imshow(img, title, file_name):
    plt.clf()
    np_img = img.cpu().numpy()
    plt.title(title)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.savefig(f"{RESULTS_DIR}/{file_name}")


def plot_loss(x, y, filename):
    plt.clf()
    plt.title(f"Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("|| D(E(x)) - x ||")
    plt.plot(x, y)
    plt.savefig(f"{RESULTS_DIR}/{filename}")


if __name__ == '__main__':
    train_test_ae()
