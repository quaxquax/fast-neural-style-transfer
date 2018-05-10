import argparse
import os

from PIL import Image
from tqdm import tqdm
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
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), # / 255.
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # MODEL
    # Define Image Transformation Network with MSE loss and Adam optimizer
    transformer = TransformerNet().to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), args.learning_rate)

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
    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.

        for batch_id, (x, _) in tqdm(enumerate(train_loader), unit='batch'):
            x = x.to(device)
            n_batch = len(x)

            optimizer.zero_grad()

            # Parse throught Image Transformation network
            y = transformer(x)
            y = normalize_batch(y)
            x = normalize_batch(x)

            # Parse through VGG layers
            features_y = vgg(y)
            features_x = vgg(x)

            # Calculate content loss
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Calculate style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            # Monitor
            if (batch_id + 1) % args.log_interval == 0:
                tqdm.write('[{}] ({})\t'
                           'content: {:.6f}\t'
                           'style: {:.6f}\t'
                           'total: {:.6f}'.format(epoch+1, batch_id+1,
                                                  agg_content_loss / (batch_id + 1),
                                                  agg_style_loss / (batch_id + 1),
                                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)))

            # Checkpoint
            if (batch_id + 1) % args.save_interval == 0:
                # eval mode
                transformer.eval().cpu()
                # TODO: Change fnst -> name of style image
                checkpoint_file = os.path.join(args.checkpoint_dir,
                                               'fnst_{}_{}.pth'.format(epoch+1, batch_id+1))

                tqdm.write('Checkpoint {}'.format(checkpoint_file))
                torch.save(transformer.state_dict(), checkpoint_file)

                # back to train mode
                transformer.to(device).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to COCO dataset')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to the style image')

    parser.add_argument('--image-size', type=int, default=256,
                        help='Size of training images')
    parser.add_argument('--content-weight', type=float, default=1e5,
                        help='Weight for content loss')
    parser.add_argument('--style-weight', type=float, default=1e10,
                        help='Weight fo style loss')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=2000)

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    main(args)
