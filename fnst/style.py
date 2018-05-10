import argparse

from PIL import Image
import torch
from torchvision import transforms

from transformer_net import TransformerNet


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content_image = Image.open(args.content)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content = transform(content_image)
    content = content.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)

        # TODO: remove saved deprecated running_* keys in InstanceNorm
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # Forward through Image Transformation Network
        out = style_model(content).cpu()

    # Save result image
    out = out.clamp(0, 255).numpy()
    # transpose (C, H, W) -> (H, W, C)
    out = out.tranpose(1, 2, 0).astype('uint8')
    out = Image.fromarray(out)
    out.save(args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, required=True,
                        help='Path to the content image')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to the result image')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the style model')
    args = parser.parse_args()
    print(args)
    main(args)
