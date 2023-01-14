import os

from PIL import Image
from tqdm import tqdm

from fcn_segmentation import FCN_Segmentation
from utils.utils_metrics import compute_mIoU, show_results


def parse_args():
    import argparse

    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou
    #   miou_mode为1代表仅仅获得预测结果
    #   miou_mode为2代表仅仅计算miou

    #   mix_type参数用于控制检测结果的可视化方式
    #   mix_type = 0的时候代表原图与生成的图进行混合
    #   mix_type = 1的时候代表仅保留生成的图
    #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
    # ---------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="model validation")
    # ---------- 验证模式的参数 ----------
    parser.add_argument(
        "--miou-mode",
        default=0,
        type=int,
        help="0, 1, 2",
    ),
    parser.add_argument(
        "--mix-type",
        default=0,
        type=int,
        help="0混合, 1仅原图, 2仅原图中的目标_扣去背景 get_miou不起作用",
    ),

    # ---------- 卷积模型的参数 ----------
    parser.add_argument(
        "--model-path",
        default="./logs/best_epoch_weights.pth",
        type=str,
    ),
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
    ),
    parser.add_argument(
        "--aux-branch",
        default=True,
        type=bool,
    ),
    parser.add_argument(
        "--num-classes",
        default=7,
        type=int,
    ),
    parser.add_argument(
        "--input-shape",
        default=[512, 512],
        type=int,
    ),
    parser.add_argument(
        "--cuda",
        default=True,
        type=bool,
    ),

    # ---------- 文件夹的位置参数 ----------
    parser.add_argument(
        "--dataset-path",
        default="../../dataset/SUIMdevkit",
        type=str,
    ),
    parser.add_argument(
        "--file-name",
        default="val.txt",
        type=str,
    ),
    parser.add_argument(
        "--save-file-dir",
        default="./miou_out_val",
        type=str,
    )

    args = parser.parse_args()
    return args


def main(args):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou
    #   miou_mode为1代表仅仅获得预测结果
    #   miou_mode为2代表仅仅计算miou
    # ---------------------------------------------------------------------------#
    miou_mode = args.miou_mode
    num_classes = args.num_classes  # 背景+物体的类别数
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = [
        "Background_waterbody",
        "Human_divers",
        "Wrecks_and_ruins",
        "Robots",
        "Reefs_and_invertebrates",
        "Fish_and_vertebrates",
        "sea_floor_and_rocks",
    ]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    SUIMdevkit_path = args.dataset_path

    image_ids = (
        open(
            os.path.join(
                SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation", args.file_name
            ),
            "r",
        )
        .read()
        .splitlines()
    )
    gt_dir = os.path.join(SUIMdevkit_path, "SUIM2022/SegmentationClass/")
    miou_out_path = args.save_file_dir
    pred_dir = os.path.join(miou_out_path, "detection-results")

    # ---------- 生成预测的mask ----------
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("💾💾💾 Load model")
        fcn_net = FCN_Segmentation(
            args.model_path,
            args.num_classes,
            args.backbone,
            args.input_shape,
            args.aux_branch,
            args.mix_type,
            args.cuda,
        )
        print("💾💾💾 Load model done")

        print("---------- Get predict result ----------")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                SUIMdevkit_path, "SUIM2022/JPEGImages/" + image_id + ".jpg"
            )
            image = Image.open(image_path)
            image = fcn_net.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("---------- Get predict result done ----------")

    # ---------- 计算预测的mask和真实的mask 混淆矩阵 ----------
    if miou_mode == 0 or miou_mode == 2:
        print("---------- Get miou ----------")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )  # 执行计算mIoU的函数
        print("---------- Get miou done ----------")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
