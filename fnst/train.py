import argparse

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from transformer_net import TransformerNet


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DATA
    # Transform and Dataloader for COCO dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        # transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), # / 255.
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # MODEL
    # Define Image Transformation Network with MSE loss and Adam optimizer
    transformer = TransformerNet().to(device)
    mse_loss = nn.MSELoss()
    optimize = optim.Adam(transformer.parameters(), args.learning_rate)

    # Precompute style features
    # 1. Extract features with VGG
    # 2. Loop through style layers to calculate the Gram Matrix

    # TRAIN
    # For each epoch:
    # 1. Parse through Image Transformation Net. Then calculate content loss
    # 2. Calculate style loss
    # 3. Log these losses and regularly save model
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to COCO dataset')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to the style image')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Size of training images')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
