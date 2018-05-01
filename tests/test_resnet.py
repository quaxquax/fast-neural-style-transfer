"""Test pre-trained ResNet50 model by classifying some test images
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fnst.resnet import resnet50

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from PIL import Image


class ResNetTest(unittest.TestCase):
    def test_classify(self):
        # Load model
        model = resnet50()
        # ResNet use batch normalization, so need to switch to eval mode in testing
        model.eval()

        # Load image data
        im_size = 224
        transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(), # / 255.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = Image.open('assets/goldfish.jpg')
        x = transform(img).float()
        with torch.no_grad():
            x = Variable(x)
        x = x.unsqueeze(0) # batch

        # Forward
        out = model(x)

        # Get index of the max
        idx = torch.argmax(out.data, 1)

        # Make sure it is a goldfish
        self.assertEqual(idx[0], 1)


if __name__ == '__main__':
    unittest.main()
