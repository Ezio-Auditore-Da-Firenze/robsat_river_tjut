import sys
import os
import argparse

# 添加整个robosat项目路径
from robosat.robosat.utils import plot

sys.path.append("F:\\PyCharmWorkSpace\\robsat_train\\robosat")
from robosat.robosat.tools.train import *
from robosat.robosat.tools.train import main


def parse_default():
    parse = argparse.ArgumentParser(description="test")
    parse.add_argument("--model", type=str, default="water_model-unet.toml", help="path to model configuration file")
    parse.add_argument("--dataset", type=str, default="water_dataset.toml", help="path to dataset configuration file")
    parse.add_argument("--checkpoint", type=str, default=False, help="path to a model checkpoint (to retrain)")
    parse.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
    parse.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    return parse.parse_args()


def test_train():
    # 此函数 解析 train 文件的过程
    global weight
    from robosat.robosat.tools.train import train
    from robosat.robosat.tools.train import validate
    from robosat.robosat.config import load_config
    from robosat.robosat.unet import UNet

    from torch.nn import DataParallel
    from robosat.robosat.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
    import collections
    from robosat.robosat.log import Log
    args = parse_default()
    print(args)
    model = load_config(args.model)
    dataset = load_config(args.dataset)
    print(dataset)
    workers = args.workers
    print(model)

    device = torch.device("cuda" if model["common"]["cuda"] else "cpu")
    print("device", device)
    if model["common"]["cuda"] and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")
    # 生成文件夹，文件夹在根目录下
    os.makedirs(model["common"]["checkpoint"], exist_ok=True)
    num_classes = len(dataset["common"]["classes"])
    print("num_classes", num_classes)
    #####################################################
    # 加载Unet模型 默认下载resnet模型，我的保存在C:\Users\Administrator/.cache\torch\checkpoints\resnet50-19c8e357.pth
    net = UNet(num_classes)
    net = DataParallel(net)
    net = net.to(device)
    print(net)
    if model["common"]["cuda"]:
        torch.backends.cudnn.benchmark = True
    ##################################################
    # 设置训练参数
    # 如果使用"CrossEntropy", "mIoU", "Focal"损失函数，必须要有weight
    try:
        weight = torch.Tensor(dataset["weights"]["values"])
    except KeyError:
        if model["opt"]["loss"] in ("CrossEntropy", "mIoU", "Focal"):
            sys.exit("Error: The loss function used, need dataset weights values")
    optimizer = Adam(net.parameters(), lr=model["opt"]["lr"])
    resume = 0
    if args.checkpoint:
        # 具体干啥不知道，默认值设置成false，就不用执行了
        pass
    if model["opt"]["loss"] == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "mIoU":
        criterion = mIoULoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Focal":
        criterion = FocalLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Lovasz":
        criterion = LovaszLoss2d().to(device)
    else:
        sys.exit("Error: Unknown [opt][loss] value !")
    #####################################################################
    # 加载数据集
    target_size = (model["common"]["image_size"],) * 2
    print("target_size", target_size)
    batch_size = model["common"]["batch_size"]
    print("batch_size", batch_size)
    # 数据集的路径
    path = dataset["common"]["dataset"]
    print("path", path)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    from robosat.robosat.transforms import (
        JointCompose,
        JointTransform,
        JointRandomHorizontalFlip,
        JointRandomRotation,
        ConvertImageMode,
        ImageToTensor,
        MaskToTensor,
    )
    from torchvision.transforms import Resize, CenterCrop, Normalize
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    from robosat.robosat.datasets import SlippyMapTilesConcatenation
    train_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "training", "images")], os.path.join(path, "training", "labels"), transform
    )
    val_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "validation", "images")], os.path.join(path, "validation", "labels"), transform
    )
    print("len train_dataset:", len(train_dataset))
    print("len val_dataset:", len(val_dataset))
    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)
    ############################################
    # 保存训练参数
    num_epochs = model["opt"]["epochs"]
    if resume >= num_epochs:
        sys.exit("Error: Epoch {} set in {} already reached by the checkpoint provided".format(num_epochs, args.model))
    history = collections.defaultdict(list)
    log = Log(os.path.join(model["common"]["checkpoint"], "log"))
    log.log("--- Hyper Parameters on Dataset: {} ---".format(dataset["common"]["dataset"]))
    log.log("Batch Size:\t {}".format(model["common"]["batch_size"]))
    log.log("Image Size:\t {}".format(model["common"]["image_size"]))
    log.log("Learning Rate:\t {}".format(model["opt"]["lr"]))
    log.log("Loss function:\t {}".format(model["opt"]["loss"]))
    if "weight" in locals():
        log.log("Weights :\t {}".format(dataset["weights"]["values"]))
    log.log("---")
    ##########################################################
    # 开始训练
    for epoch in range(resume, num_epochs):
        log.log("Epoch: {}/{}".format(epoch + 1, num_epochs))

        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        log.log(
            "Train    loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                train_hist["loss"],
                train_hist["miou"],
                dataset["common"]["classes"][1],
                train_hist["fg_iou"],
                train_hist["mcc"],
            )
        )

        for k, v in train_hist.items():
            history["train " + k].append(v)
        val_hist = validate(val_loader, num_classes, device, net, criterion)
        log.log(
            "Validate loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                val_hist["loss"], val_hist["miou"], dataset["common"]["classes"][1], val_hist["fg_iou"], val_hist["mcc"]
            )
        )

        for k, v in val_hist.items():
            history["val " + k].append(v)

        visual = "2.7-history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
        plot(os.path.join(model["common"]["checkpoint"], visual), history)

        if (epoch + 1) % 2 == 0:
            checkpoint = "2.7-checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, num_epochs)
            states = {"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(states, os.path.join(model["common"]["checkpoint"], checkpoint))



if __name__ == '__main__':
    args = parse_default()
    # main(args)
    test_train()
