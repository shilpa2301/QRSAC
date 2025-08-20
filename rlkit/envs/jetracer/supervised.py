import torch
from torch.utils.data import Dataset, Subset
import torch.nn as nn
import torchvision.transforms as transforms

import os
import pandas as pd
from torch.utils.data import DataLoader
import cv2
import numpy as np

import optuna

from torchvision.models import resnet18 as VisionModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


learning_rate = 1e-3
batch_size = 64
epochs = 1000
num_stacked_images = 3


augmentation_transformer = transforms.Compose([
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=.9, hue=.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast()
    ]),
    transforms.Resize((224, 224), antialias=False),
    transforms.Lambda(lambda x: x / 255.)
])


def stack_transformer(image): return np.transpose(image, (1, 0))     # Torch is CxWxH, cv2 is HxW


eval_transformer = transforms.Compose([
    transforms.Resize((224, 224), antialias=False),
    transforms.Lambda(lambda x: x / 255.)
])


def stack_images(idx, num_stacked, labels_csv, directory):
    images = list()
    for i in range(num_stacked):
        if idx - i < 0:
            image = np.copy(images[0])
        else:
            img_path = os.path.join(directory, labels_csv.iloc[idx-i, 0])
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = stack_transformer(image)
        images.append(image)
    stacked = np.stack(images, axis=0)
    return stacked


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dirs):
        combined_index = 0
        labels = dict()
        images = dict()
        for dataset_dir in img_dirs:
            labels_csv = pd.read_csv(os.path.join(dataset_dir, annotations_file))
            for i in range(len(labels_csv)):
                label = labels_csv.iloc[i, 1]   # 1 is the col for steering inputs

                # if i == 0:
                #     speed = 0.0
                # else:
                #     x2 = labels_csv.iloc[i, 3]
                #     y2 = labels_csv.iloc[i, 4]
                #     t2 = labels_csv.iloc[i, 8]
                #     x1 = labels_csv.iloc[i-1, 4]
                #     y1 = labels_csv.iloc[i-1, 4]
                #     t1 = labels_csv.iloc[i-1, 8]
                #     speed = ((x2-x1)**2 + (y2-y1)**2)**(1/2) / (t2-t1)
                # label = speed

                label = torch.FloatTensor(np.expand_dims(np.float32(label), axis=0)).to(device)
                labels[combined_index] = label

                img = stack_images(i, num_stacked_images, labels_csv, dataset_dir)
                img = torch.Tensor(img).to(device)     # TODO load gpu
                images[combined_index] = img

                combined_index += 1

        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        restack = list()
        for i in range(img.shape[0]):
            imgi = torch.unsqueeze(img[i, :, :], dim=0)
            imgi = augmentation_transformer(imgi)
            restack.append(imgi)
        img = torch.vstack(restack)

        return img, self.labels[idx]


def train_loop(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    test_loss /= num_batches
    print(f"Train Error: Avg loss: {test_loss:>8f}")
    return test_loss


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f}")
    return test_loss


def transformer(image):
    # scale_percent = 25  # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    dim = (224, 224)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    image = image.astype(np.float32)
    image /= 255.
    image = torch.FloatTensor(image)
    return image


def load_data():
    data = CustomImageDataset('labels.csv', ['2023-09-30_15:50:08', '2023-09-30_15:33:10'])
    # split_index = int(len(data)*7/8)
    # training_data = Subset(data, range(0, split_index))
    # test_data = Subset(data, range(split_index+1, len(data)))

    # train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    test_dataloader = None
    return train_dataloader, test_dataloader


def train_model():
    train_dataloader, test_dataloader = load_data()

    model = VisionModel(num_classes=1)
    model.to(device)
    loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss, best_epoch = np.inf, 0
    for t in range(epochs):
        print(f"Epoch {t+1}--------------------")
        test_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        if test_dataloader is not None:
            test_loss = test_loop(test_dataloader, model, loss_fn)
        if t % 5 == 0 and t != 0:
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = t
            print(best_epoch, best_loss)
            torch.save(model, 'ctrl_model_'+str(t)+'.pth')


def main():
    train_model()


if __name__ == '__main__':
    main()
