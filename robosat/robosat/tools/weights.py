import os
import argparse
import sys

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from robosat.robosat.config import load_config
from robosat.robosat.datasets import SlippyMapTiles
from robosat.robosat.transforms import ConvertImageMode, MaskToTensor

sys.path.append("F:\\PyCharmWorkSpace\\robsat_train")
def add_parser(subparser):
    parser = subparser.add_parser(
        "weights", help="computes class weights on dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", type=str, default="F:\\PyCharmWorkSpace\\robsat_train\\water_dataset.toml", help="path to dataset configuration file")

    parser.set_defaults(func=main)

def parse_default():
    parse = argparse.ArgumentParser(description="test")
    parse.add_argument("--dataset", type=str, default="F:\\PyCharmWorkSpace\\robsat_train\\water_dataset.toml", help="path to dataset configuration file")
    return parse.parse_args()

def main():
    args = parse_default()
    print(args)
    dataset = load_config(args.dataset)

    path = dataset["common"]["dataset"]
    num_classes = len(dataset["common"]["classes"])

    train_transform = Compose([ConvertImageMode(mode="P"), MaskToTensor()])

    train_dataset = SlippyMapTiles(os.path.join(path, "training", "labels"), transform=train_transform)

    n = 0
    counts = np.zeros(num_classes, dtype=np.int64)

    loader = DataLoader(train_dataset, batch_size=1)
    for images, tile in tqdm(loader, desc="Loading", unit="image", ascii=True):
        image = torch.squeeze(images)

        image = np.array(image, dtype=np.uint8)
        n += image.shape[0] * image.shape[1]
        counts += np.bincount(image.ravel(), minlength=num_classes)

    assert n > 0, "dataset with masks must not be empty"

    # Class weighting scheme `w = 1 / ln(c + p)` see:
    # - https://arxiv.org/abs/1707.03718
    #     LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    # - https://arxiv.org/abs/1606.02147
    #     ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation

    probs = counts / n
    weights = 1 / np.log(1.02 + probs)

    weights.round(6, out=weights)
    print(weights.tolist())

if __name__ == '__main__':
    # args = parse_default()
    main()
    # test_train()