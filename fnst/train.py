import argparse

from PIL import Image
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from transformer_net import TransformerNet
from vgg import VGG16


def normalize_batch(batch):
    """Normalize batch using ImageNet mean and std
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std


def gram_matrix(y):
    (batch, channel, height, width) = y.size()
    features = y.view(batch, channel, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (channel * height * width)
    return gram


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

    # Pretrained VGG
    vgg = VGG16(requires_grad=False).to(device)

    # FEATURES
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load the style image
    style = Image.open(args.style)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # Compute the style features
    features_style = vgg(normalize_batch(style))

    # Loop through VGG style layers to calculate Gram Matrix
    gram_style = [gram_matrix(y) for y in features_style]

    # TRAIN
    # For each epoch:
    # 1. Parse through Image Transformation Net. Then calculate content loss
    # 2. Calculate style loss
    # 3. Log these losses and regularly save model


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