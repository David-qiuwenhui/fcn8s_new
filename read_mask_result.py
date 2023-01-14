'''
@author: qiuwenhui
@Software: VSCode
@Time: 2023-01-11 15:45:46
'''
import json
import os
from PIL import Image
import numpy as np


def main():
    root = "./miou_out_val/detection-results"
    file_name = "d_r_47_.png"
    image_path = os.path.join(root, file_name)
    save_path = os.path.join(root, file_name.split(".")[0] + "new" + ".png")
    
    palette_path = "./palette_suim.json"
    with open(palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    image = Image.open(image_path)
    mask = np.asarray(image)
    height, width = mask.shape
    total_pixels = height * width
    print(f"shape: {height}, {width}")
    print(f"像素的总数量为:{total_pixels}")

    memo_list = []
    memo_dict = {}
    for h in range(height):
        for w in range(width):
            if mask[h][w] not in memo_list:
                memo_list.append(mask[h][w])
                memo_dict[mask[h][w]] = 1
            else:
                memo_dict[mask[h][w]] += 1

    print(memo_list)
    print(memo_dict)
    for key, value in memo_dict.items():
        print(f"像素值{key}, 整张图片占比{value/total_pixels: .2%}")

    new_mask = Image.fromarray(mask.copy())
    new_mask.putpalette(palette, rawmode="BGR")
    new_mask.save(save_path)
    
    




if __name__ == "__main__":
    main()