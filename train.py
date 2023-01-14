import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.fcn_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from utils.utils import time_synchronized

from nets.fcn_model import fcn_resnet50, fcn_resnet101

"""
    如果格式有误，参考：https://github.com/bubbliiiing/segmentation-format-fix
    调参是一门蛮重要的学问，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
    这些都是经验上，只能靠各位同学多查询资料和自己试试了。
"""


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")
    # ---------- 数据集超参数 ----------
    parser.add_argument(
        "--data-path",
        default="../../dataset/SUIMdevkit",
        type=str,
        help="dataset root",
    )
    parser.add_argument(
        "--input-size", default=512, type=int, help="the size of input image"
    )

    # ---------- 卷积模型超参数 ----------
    parser.add_argument("--backbone", default="resnet50", help="resnet50, resnet101")
    # parser.add_argument("--downsample-factor", default=8, type=int, help="8, 16")
    parser.add_argument("--aux-branch", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--num-classes", default=7, type=int)
    parser.add_argument("--pretrained-model", default=False, type=bool)
    parser.add_argument("--pretrained-backbone", default=False, type=bool)
    parser.add_argument(
        "--model-path",
        default="",
        help="model weights path",
    )
    parser.add_argument(
        "--backbone-path",
        default="",
        help="backbone weights path",
    )

    # ---------- 硬件的超参数 ----------
    parser.add_argument("--cuda", default=True, type=bool, help="use cuda")
    parser.add_argument(
        "--amp",
        default=True,
        type=bool,
        help="Use torch.cuda.amp for mixed precision training",
    )

    # ---------- 训练Epoch和Batch size超参数 ----------
    parser.add_argument("--freeze-train", default=False, type=bool)
    parser.add_argument("--freeze-batch-size", default=16, type=int)
    parser.add_argument("--unfreeze-batch-size", default=16, type=int)
    parser.add_argument(
        "--init-epoch", default=0, type=int, metavar="N", help="init epoch"
    )
    parser.add_argument(
        "--freeze-epochs",
        default=0,
        type=int,
        metavar="N",
        help="number of freeze epochs to train",
    )
    parser.add_argument(
        "--unfreeze-epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of unfreeze epochs to train",
    )

    # ---------- 训练的优化器超参数 ----------
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument(
        "--init-lr", default=1e-2, type=float, help="initial learning rate"
    )
    parser.add_argument("--lr-decay-type", default="cos", type=str)
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    # ---------- 损失函数的超参数 ----------
    parser.add_argument("--dice-loss", default=False, type=bool)
    parser.add_argument("--focal-loss", default=False, type=bool)

    # ---------- 模型验证和保存的超参数 ----------
    parser.add_argument("--eval-flag", default=True, type=bool)
    parser.add_argument("--eval-period", default=5, type=int)
    parser.add_argument("--save-dir", default="./logs", type=str, help="save directory")
    parser.add_argument("--save-period", default=5, type=int, help="save frequency")

    args = parser.parse_args()
    return args


def create_model(aux, num_classes, pretrain=False):
    model = fcn_resnet50(aux, num_classes, pretrain_backbone=pretrain)

    if pretrain:
        # fcn resnet50
        weights_dict = torch.load(
            "./pre_weights/fcn_resnet50_coco.pth", map_location="cpu"
        )

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景，20classes+1background)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=weights_dict, strict=False
        )
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    Cuda = args.cuda
    distributed = False
    sync_bn = False  # 是否使用sync_bn，DDP模式多卡可用
    fp16 = args.amp  # 使用混合精度训练 可减少约一半的显存
    num_classes = args.num_classes  # 物体类别+背景类别
    backbone = args.backbone
    pretrained_backbone = args.pretrained_backbone
    backbone_path = args.backbone_path
    model_path = args.model_path
    input_shape = [args.input_size, args.input_size]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 4e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 4e-3，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 4e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 4e-3，weight_decay = 1e-4。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合语义分割，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在120-300之间调整。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = args.init_epoch
    Freeze_Train = args.freeze_train
    Freeze_Epoch = args.freeze_epochs
    Freeze_batch_size = args.freeze_batch_size
    UnFreeze_Epoch = args.unfreeze_epochs
    Unfreeze_batch_size = args.unfreeze_batch_size

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=4e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = args.init_lr
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=4e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = args.optimizer
    momentum = args.momentum
    weight_decay = args.weight_decay
    lr_decay_type = args.lr_decay_type  # 使用到的学习率下降方式，可选的有'step'、'cos'
    save_period = args.save_period  # 多少个epoch保存一次权值，默认每个世代都保存
    save_dir = args.save_dir  # 权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = args.eval_flag
    eval_period = args.eval_period

    SUIMdevkit_path = args.data_path  # 数据集路径
    # ------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ------------------------------------------------------------------#
    dice_loss = args.dice_loss
    # ------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ------------------------------------------------------------------#
    focal_loss = args.focal_loss
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    aux_branch = args.aux_branch
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = min([os.cpu_count(), Freeze_batch_size, Unfreeze_batch_size, 8])

    #   设置用到的显卡
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training..."
            )
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # ---------- 实例化卷积神经网络模型 ----------
    if backbone == "resnet50":
        model = fcn_resnet50(
            aux_branch,
            num_classes,
            model_path,
            backbone_path,
        )
    elif backbone == "resnet101":
        model = fcn_resnet101(
            aux_branch,
            num_classes,
            model_path,
            backbone_path,
        )

    # 模型的参数初始化
    if not pretrained_backbone:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            print("Load weights {}.".format(model_path))
        #   根据预训练权重的Key和模型的Key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #   显示没有匹配上的Key
        if local_rank == 0:
            print(
                "\nSuccessful Load Key:",
                str(load_key)[:500],
                "……\nSuccessful Load Key Num:",
                len(load_key),
            )
            print(
                "\nFail To Load Key:",
                str(no_load_key)[:500],
                "……\nFail To Load Key num:",
                len(no_load_key),
            )
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #  记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
        )
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape)
    else:
        loss_history = None
    # 设置混合精度训练
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    #   多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # 多卡平行运行
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #   读取数据集对应的txt
    with open(
        os.path.join(SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation/train.txt"), "r"
    ) as f:
        train_lines = f.readlines()
    with open(
        os.path.join(SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation/val.txt"), "r"
    ) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            aux_branch=aux_branch,
            num_classes=num_classes,
            Init_Epoch=Init_Epoch,
            Freeze_Train=Freeze_Train,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            optimizer_type=optimizer_type,
            lr_decay_type=lr_decay_type,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dice_loss=dice_loss,
            focal_loss=focal_loss,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
            save_period=save_period,
            save_dir=save_dir,
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4  # 1.5e4=15000
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"
                % (optimizer_type, wanted_step)
            )
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    UnFreeze_flag = False
    #   冻结模型的参数
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # ---------- 配置训练的优化器 ----------
    #  判断当前batch_size，自适应调整学习率
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(
        max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
    )

    # 根据optimizer_type选择优化器
    optimizer = {
        "adam": optim.Adam(
            params=model.parameters(),
            lr=Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "sgd": optim.SGD(
            params=model.parameters(),
            lr=Init_lr_fit,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        ),
    }[optimizer_type]

    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
    )
    # 判断每一个epoch的长度
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小或batch size过大，无法继续进行训练，请扩充数据集。")

    # ---------- 实例化训练集和测试集 ----------
    train_dataset = SegmentationDataset(
        train_lines, input_shape, num_classes, train=True, dataset_path=SUIMdevkit_path
    )
    val_dataset = SegmentationDataset(
        val_lines, input_shape, num_classes, train=False, dataset_path=SUIMdevkit_path
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # 将训练数据载入内存
    gen = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
        sampler=train_sampler,
    )
    gen_val = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
        sampler=val_sampler,
    )

    #   记录eval的map曲线
    if local_rank == 0:
        eval_callback = EvalCallback(
            net=model,
            input_shape=input_shape,
            num_classes=num_classes,
            image_ids=val_lines,
            dataset_path=SUIMdevkit_path,
            log_dir=log_dir,
            cuda=Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )
    else:
        eval_callback = None

    # ---------- 开始模型训练 ----------
    start_time = time_synchronized()
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        #   如果模型有冻结学习部分 则解冻，并设置参数
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            #   判断当前batch_size，自适应调整学习率
            nbs = 16
            lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
            lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
            Init_lr_fit = min(
                max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
            )
            Min_lr_fit = min(
                max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
            )
            #   获得学习率下降的公式
            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
            )

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = DataLoader(
                dataset=train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=seg_dataset_collate,
                sampler=train_sampler,
            )
            gen_val = DataLoader(
                dataset=val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=seg_dataset_collate,
                sampler=val_sampler,
            )

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            aux_branch,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            dice_loss,
            focal_loss,
            cls_weights,
            num_classes,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
        )

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
    total_time = time_synchronized() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("*********** 🚀🚀🚀🚀 Finish training! 🚀🚀🚀🚀 ***********")
    print("*********** training time {} ***********".format(total_time_str))


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
