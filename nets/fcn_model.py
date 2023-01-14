from collections import OrderedDict
from typing import Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


class FeatureFuse(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*layers)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):  # set() 函数创建一个集合，可用来进行关系测试
            raise ValueError(
                "return_layers are not present in model"
            )  # issubset() 方法用于判断集合的所有元素是否都包含在指定集合中
        orig_return_layers = return_layers
        return_layers = {
            str(k): str(v) for k, v in return_layers.items()
        }  # Python字典items()方法以列表返回视图对象，是一个可遍历的key/value对

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()  # 有序字典
        for name, module in model.named_children():  # backbone的子模块（模块名称，模块数据）
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()  # 创建有序字典保存主分类器和aux辅助分类器的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    __constants__ = ["aux_classifier"]

    def __init__(self, backbone, num_classes, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        self.layersfuse1 = FeatureFuse(1024, num_classes)
        self.layersfuse2 = FeatureFuse(512, num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]  # input_tensor height and weight (H, W)

        # contract: features is a dict of tensors
        features = self.backbone(
            x
        )  # x(B,3,H,W) -> features OrderedDict: "down8"(B,512,H/8,H/8) "down16(B,1024,H/16,H/16) "out"(B,2048,H/32,H/32)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)  # 主分类器classifier输出预测分类结果 (B, N, H/32, W/32)

        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(
            input=x,
            size=features["down16"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # x(B,N,H/16,W/16)  align_corners=False使用origin_image边缘数值填充新的图像边缘，能保证整数倍的上下采样

        # 深浅层特征融合1
        x = x + self.layersfuse1(features["down16"])
        x = F.interpolate(
            input=x,
            size=features["down8"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # x(B,N,H/8,W/8)

        # 深浅层特征融合2
        x = x + self.layersfuse2(features["down8"])
        x = F.interpolate(
            input=x, size=input_shape, mode="bilinear", align_corners=False
        )  # x(B,N,H,W)

        result["out"] = x
        # 启用辅助分类器
        if self.aux_classifier is not None:
            x = features["aux"]  # backbone["aux"]输出特征 (bs, 1024, 60, 60)
            x = self.aux_classifier(x)  # x(B,N,60,60)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(
                x, size=input_shape, mode="bilinear", align_corners=False
            )  # (B,N,480,480)
            result["aux"] = x  # OrderedDict{'out':tensor, 'aux':tensor}
            # 'out':Tensor(bs,N,480,480) 'aux':Tensor(bs,N,480,480)
            return (result["out"], result["aux"])

        return result["out"]


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):  # in_channels=1024, channels=21
        # 网络中间层的通道数
        inter_channels = in_channels // 4  # aux: 1024//4=256 out: 2048//4=512
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1),
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(
    aux,
    num_classes=7,
    pretrained_backbone=False,
    backbone_path="",
):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    # ---------- backbone ----------
    # 两次特征融合的FCN-8S 更改下采样倍率
    backbone = resnet50(replace_stride_with_dilation=[False, False, False])
    # backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    if pretrained_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
    # ------------------------------

    aux_inplanes = 1024
    out_inplanes = 2048
    return_layers = {"layer4": "out"}
    # 若使用辅助分类器
    if aux:
        return_layers["layer3"] = "aux"  # dict:2 {'layer4': 'out', 'layer3': 'aux'}

    inter_layers = {"layer2": "down8", "layer3": "down16"}
    return_layers.update(inter_layers)

    # 重构FCN的backbone主干特征提取网络
    backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layers)

    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)
    else:
        aux_classifier = None

    # 主分类器分支
    classifier = FCNHead(out_inplanes, num_classes)
    model = FCN(backbone, num_classes, classifier, aux_classifier)
    return model


def fcn_resnet101(
    aux,
    num_classes=7,
    pretrained_backbone=False,
    backbone_path="",
):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    # ---------- backbone ----------
    # 两次上下文特征融合的FCN-8S 更改下采样倍率
    backbone = resnet101(replace_stride_with_dilation=[False, False, False])
    if pretrained_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
    # ------------------------------

    out_inplanes = 2048
    aux_inplanes = 1024
    return_layers = {"layer4": "out"}
    # 若使用辅助分类器
    if aux:
        return_layers["layer3"] = "aux"

    inter_layers = {"layer2": "down8", "layer3": "down16"}
    return_layers.update(inter_layers)

    # 重构FCN的backbone主干特征提取网络
    backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layers)

    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)
    else:
        aux_classifier = None

    # 主分类器分支
    classifier = FCNHead(out_inplanes, num_classes)
    model = FCN(backbone, num_classes, classifier, aux_classifier)
    return model
