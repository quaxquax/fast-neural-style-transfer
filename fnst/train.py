import argparse


def main(args):
    # DATA
    # Transform and Dataloader for COCO dataset

    # MODEL
    # Define Image Transformation Network with MSE loss and Adam optimizer

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
    args = parser.parse_args()
    main(args)
