import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import yaml

DATASET_DIR = "./dataset"

# Hyper Parameters
NUM_TEST_IMAGES = 1000
TEST_BATCH_SIZE = 32
ADAM_BETAS = (0.9, 0.999)
NUM_EPOCHS = 10


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class AE(nn.Module):

    def __init__(self, config):
        super(AE, self).__init__()
        relu_slope = float(config['relu_slope'])
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.MaxPool2d(2),  # 128 x 128 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(32, 32, 3, padding="same"),
            nn.MaxPool2d(2),  # 64 x 64 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(32, 64, 3, padding="same"),
            nn.MaxPool2d(2),  # 32 x 32 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(64, 64, 3, padding="same"),
            nn.MaxPool2d(2),  # 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(64, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 8 x 8 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 4 x 4 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.MaxPool2d(2),  # 2 x 2 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(128, 256, 3, padding="same"),
            nn.MaxPool2d(2),  # 1 x 1 x 256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Flatten(),  # 1 x 256
        )

        self.decoder = nn.Sequential(
            View((-1, 256, 1, 1)),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),  # 16 x 16 x 64

            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),  # 32 x 32 x 64

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),  # 64 x 64 x 32

            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),  # 256 x 256 x 32

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(relu_slope, inplace=True),  # 256 x 256 x 3

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


def train_test_ae(config):
    lr = float(config['lr'])
    weight_decay = float(config['weight_decay'])
    batch_size = int(config['batch_size'])
    results_dir = config['results_dir']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae = AE(config)
    ae.to(device)
    ae.apply(weights_init)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr, betas=ADAM_BETAS, weight_decay=weight_decay)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    test_dataset = Subset(dataset, np.arange(NUM_TEST_IMAGES))
    train_dataset = Subset(dataset, np.arange(NUM_TEST_IMAGES, len(dataset)))
    test_dataloader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    train_loss_arr = []
    for epoch in range(config['num_epochs']):
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
                imshow(torchvision.utils.make_grid(input_images), "input", f"{results_dir}/train_input_{epoch}_{i}.pdf")
                imshow(torchvision.utils.make_grid(output_images), "output", f"{results_dir}/train_output_{epoch}_{i}.pdf")

        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, start=1):
                input_images = data[0].to(device)
                output_images = ae(input_images)
                loss = loss_fn(input_images, output_images)
                test_loss += loss.item()
                if i % 10 == 0:
                    imshow(torchvision.utils.make_grid(input_images), "input", f"{results_dir}/test_input_{epoch}_{i}.pdf")
                    imshow(torchvision.utils.make_grid(output_images), "output", f"{results_dir}/test_output_{epoch}_{i}.pdf")
            print(f'Epoch {epoch} average test loss: {test_loss / len(test_dataloader)}')


def imshow(img, title, file_path):
    plt.clf()
    np_img = img.cpu().numpy()
    plt.title(title)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.savefig(file_path)


def plot_loss(x, y, file_path):
    plt.clf()
    plt.title(f"Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("|| D(E(x)) - x ||")
    plt.plot(x, y)
    plt.savefig(file_path)


def load_conf():
    with open("config.yaml", "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    conf = load_conf()
    train_test_ae(conf)
