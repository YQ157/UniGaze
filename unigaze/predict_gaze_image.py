import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from infer_runtime import UniGazeRuntime
from tqdm import tqdm

# --- 设置预训练模型下载目录 ---
cache_dir = "pretrained_cache"
os.environ["TORCH_HOME"] = os.path.abspath(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

# --- 本地模型文件路径 ---
DEFAULT_CKPT_FILE = "logs/unigaze_b16_joint.pth.tar"  # 请替换为你本地模型文件的路径
DEFAULT_CFGS = "configs/model/mae_b_16_gaze.yaml"  # 配置文件路径，确保这个文件存在
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_cfg_abs(cfg_str: str) -> Path:
    p = Path(cfg_str)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {p}")
    p2 = (Path.cwd() / p).resolve()
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Config not found. Tried: {p2}")


def get_runtime(
    cfg_abs_str: str, ckpt_path: str, device: str = "cpu"
) -> UniGazeRuntime:
    return UniGazeRuntime(cfg_abs_str, ckpt_path, device=device)


# --- 辅助函数 ---
def process_image_with_unigaze(image_path, runtime):
    image_original = cv2.imread(image_path)
    if image_original is None:
        return None

    # 使用 UniGaze 进行视点估计
    output = runtime.predict_image(image_original)

    # 返回结果
    return output


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_dir", default="./output_images")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # 加载 UniGaze 模型
    cfg_abs = resolve_cfg_abs(DEFAULT_CFGS)
    runtime = get_runtime(str(cfg_abs), DEFAULT_CKPT_FILE, device=str(device))

    # 输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取输入图片路径
    image_paths = (
        [args.input_path]
        if not os.path.isdir(args.input_path)
        else sorted(glob.glob(os.path.join(args.input_path, "*.[jp][pn]g")))
    )

    # 处理每一张图片
    for image_path in tqdm(image_paths, desc="Processing images"):
        output_image = process_image_with_unigaze(image_path, runtime)
        if output_image is not None:
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, output_image)

    print("处理完成！")
