from robosat.robosat.tools.train import *
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
from robosat.robosat.datasets import BufferedSlippyMapDirectory
from robosat.robosat.colors import continuous_palette_for_color
from torchvision.transforms import Compose, Normalize
from robosat.robosat.colors import make_palette
from robosat.robosat.tiles import tiles_from_slippy_map
import cv2 as cv


def parse_default():
    parse = argparse.ArgumentParser(description="test")
    parse.add_argument("--batch_size", type=int, default=1, help="images per batch")
    parse.add_argument("--checkpoint", type=str, default="D:\\BaiduNetdiskDownload\\water_dataset\\checkpoint\\checkpoint-00040-of-00040.pth",  help="model checkpoint to load")
    parse.add_argument("--overlap", type=int, default=32, help="tile pixel overlap to predict on")
    parse.add_argument("--tile_size", type=int, default=448,  help="tile size for slippy map tiles")
    parse.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    parse.add_argument("--tiles", type=str, default="D:\\BaiduNetdiskDownload\\water_dataset\\test\\test-data", help="directory to read slippy map image tiles from")
    parse.add_argument("--probs", type=str, default="D:\\BaiduNetdiskDownload\\water_dataset\\test\\test-predict", help="directory to save slippy map probability masks to")
    # parse.add_argument("--masks", type=str, default="F:\\PyCharmWorkSpace\\robsat_train\\cat_mask",help="slippy map directory to save masks to")
    parse.add_argument("--weights", type=float, nargs="+", help="weights for weighted average soft-voting")
    parse.add_argument("--model", type=str, default="water_model-unet.toml", help="path to model configuration file")
    parse.add_argument("--dataset", type=str,default="water_dataset.toml", help="path to dataset configuration file")
    return parse.parse_args()

def predict_pic():
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    cuda = model["common"]["cuda"]

    device = torch.device("cuda" if cuda else "cpu")

    def map_location(storage, _):
        return storage.cuda() if cuda else storage.cpu()

    if cuda and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    num_classes = len(dataset["common"]["classes"])

    # https://github.com/pytorch/pytorch/issues/7178
    chkpt = torch.load(args.checkpoint, map_location=map_location)

    net = UNet(num_classes).to(device)
    net = nn.DataParallel(net)

    if cuda:
        torch.backends.cudnn.benchmark = True

    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=args.tile_size, overlap=args.overlap)
    assert len(directory) > 0, "at least one tile in dataset"

    loader = DataLoader(directory, batch_size=args.batch_size, num_workers=args.workers)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))
                # we predicted on buffered tiles; now get back probs for original image
                prob = directory.unbuffer(prob)

                # Quantize the floating point probabilities in [0,1] to [0,255] and store
                # a single-channel `.png` file with a continuous color palette attached.

                assert prob.shape[0] == 2, "single channel requires binary model"
                assert np.allclose(np.sum(prob, axis=0), 1.), "single channel requires probabilities to sum up to one"
                foreground = prob[1:, :, :]

                anchors = np.linspace(0, 1, 256)
                quantized = np.digitize(foreground, anchors).astype(np.uint8)

                palette = continuous_palette_for_color("blue", 256)

                out = Image.fromarray(quantized.squeeze(), mode="P")
                # cv.imshow("predict",palette)
                out.putpalette(palette)

                os.makedirs(os.path.join(args.probs, str(z), str(x)), exist_ok=True)
                path = os.path.join(args.probs, str(z), str(x), str(y) + ".png")

                out.save(path, optimize=True)
        return
def mask_pic():
    if args.weights and len(args.probs) != len(args.weights):
        sys.exit("Error: number of slippy map directories and weights must be the same")

    tilesets = map(tiles_from_slippy_map, args.probs)
    # print(args.probs)
    ############################################################################
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])
    # directory = BufferedSlippyMapDirectory(args.probs, transform=transform, size=args.tile_size, overlap=args.overlap)
    # assert len(directory) > 0, "at least one tile in dataset"
    # loader = DataLoader(directory, batch_size=args.batch_size, num_workers=args.workers)
    ##########################################################################################
    #list(zip(*tilesets))
    for tileset in tqdm(list(zip(*tilesets)), desc="Masks", unit="tile", ascii=True):
        tiles = [tile for tile, _ in tileset]
        paths = [path for _, path in tileset]
        #
        # assert len(set(tiles)), "tilesets in sync"
        x, y, z = tiles[0]
        # Un-quantize the probabilities in [0,255] to floating point values in [0,1]
        anchors = np.linspace(0, 1, 256)
        def load(path):
            # Note: assumes binary case and probability sums up to one.
            # Needs to be in sync with how we store them in prediction.

            quantized = np.array(Image.open(path).convert("P"))

            # (512, 512, 1) -> (1, 512, 512)
            foreground = np.rollaxis(np.expand_dims(anchors[quantized], axis=0), axis=0)
            background = np.rollaxis(1. - foreground, axis=0)

            # (1, 512, 512) + (1, 512, 512) -> (2, 512, 512)
            return np.concatenate((background, foreground), axis=0)

        probs = [load(path) for path in paths]

        mask = softvote(probs, axis=0, weights=args.weights)
        mask = mask.astype(np.uint8)

        palette = make_palette("denim", "blue")
        out = Image.fromarray(mask, mode="P")
        out.putpalette(palette)

        os.makedirs(os.path.join(args.masks, str(z), str(x)), exist_ok=True)

        path = os.path.join(args.masks, str(z), str(x), str(y) + ".png")
        out.save(path, optimize=True)

def softvote(probs, axis=0, weights=None):
    """Weighted average soft-voting to transform class probabilities into class indices.

    Args:
      probs: array-like probabilities to average.
      axis: axis or axes along which to soft-vote.
      weights: array-like for weighting probabilities.

    Notes:
      See http://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    """

    return np.argmax(np.average(probs, axis=axis, weights=weights), axis=axis)
    return



if __name__ == '__main__':
    args = parse_default()
    # main(args)
    predict_pic()# 获得预测概率
    #mask_pic()# 形成mask遮罩

