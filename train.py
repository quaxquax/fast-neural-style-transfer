import argparse


def main(args):
    # DATA
    # Get the target style image
    # Get the COCO train images

    # BUILDING GRAPH
    # Precompute style features
    # 1. Extract features with the VGG network
    # 2. Loop through the Style Layers to calculate the Gram Matrix

    # Precompute content features
    # 1. Extract features with the VGG network
    # 2. Calculate content features by parsing through Image Transform Network

    # Loss
    # 1. Calculate feature reconstruction loss
    # 2. Calculate style reconstruction loss
    # 3. Total Variation Regularization

    # Optimize (Adam)

    # TRAIN
    # 1. Loop through batch of images
    # 2. Run session for optimizing (train step),
    # log informations of the loss (style + content + tv as well) and the y_hat
    # 3. Frequently, load checkpoints and forward the test image to get the styled result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
