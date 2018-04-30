"""Test pre-trained VGG16 model by classifying some test images
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fnst.vgg import vgg16

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image


class VGGTest(unittest.TestCase):
    def test_classify(self):
        # Load model
        model = vgg16()

        # Load image data
        im_size = 224
        transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor() # / 255.
        ])

        img = Image.open('assets/goldfish.jpg')
        x = transform(img).float()
        x = Variable(x, volatile=True)
        x = x.unsqueeze(0) # batch

        # Forward
        out = model(x)
        # Softmax
        out = F.softmax(out, dim=1)

        # Get index of the max
        _, idx = torch.max(out.data, 1)

        # Make sure it is a goldfish
        self.assertEqual(idx[0], 1)


if __name__ == '__main__':
    unittest.main()
