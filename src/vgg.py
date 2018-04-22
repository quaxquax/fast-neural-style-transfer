import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        pass
        # Input (batch_size, width, height, channels)

        # 2 Conv2D 3x3 (stride 1), SAME, out 64, relu
        # 1 MaxPool 2x2 (stride 2)nn

        # 3 Conv2D 3x3 (stride 1), SAME, out 256, relu
        # 1 Maxpool 2x2 (stride 2)

        # 3 Conv2D 3x3 (stride 1), SAME, out 512, relu
        # 1 Maxpool 2x2 (stride 2)

        # 3 Conv2D 3x3 (stride 1), SAME, out 512, relu
        # 1 Maxpool 2x2 (stride 2)

        # Flatten
        # 2 FC, out 4096, Dropout 0.5, relu
        # 1 FC, out 1000, softmax
