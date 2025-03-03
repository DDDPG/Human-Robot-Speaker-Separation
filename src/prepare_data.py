import csv
import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset(
    datapath,
    savepath,
    n_spks=2,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    skip_prep=False,
):
    """
    准备数据集，并按照指定比例划分训练集、验证集和测试集。

    Arguments:
    ----------
        datapath (str): 数据集路径
        savepath (str): CSV文件保存路径
        n_spks (int): 说话人数量
        train_ratio (float): 训练集比例 (0-1)
        valid_ratio (float): 验证集比例 (0-1)
        test_ratio (float): 测试集比例 (0-1)
        random_seed (int): 随机种子
        skip_prep (bool): 是否跳过数据准备
    """
    if skip_prep:
        return
    
    # 确保比例之和为1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于1"
    
    if "HRSP2mix" in datapath:
        # 读取所有metadata文件
        metadata_list = []
        sample_paths = []
        idx = 0
        while True:
            sample_path = os.path.join(datapath, f"sample_{idx:04d}")
            metadata_path = os.path.join(sample_path, "metadata.json")
            if not os.path.exists(metadata_path):
                # print(f"metadata path: {(metadata_path)}")
                break
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
                sample_paths.append(sample_path)
            idx += 1

        if len(metadata_list) == 0:
            raise ValueError("No metadata found")
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 划分数据集
        train_valid_paths, test_paths = train_test_split(
            sample_paths,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        train_paths, valid_paths = train_test_split(
            train_valid_paths,
            test_size=valid_ratio/(train_ratio + valid_ratio),
            random_state=random_seed
        )
        
        # 创建数据集目录结构
        for set_type in ["train", "valid", "test"]:
            for folder in ["mixture", "source1", "source2"]:
                os.makedirs(os.path.join(datapath, set_type, folder), exist_ok=True)
        
        # 移动文件到对应目录
        path_sets = {
            "train": train_paths,
            "valid": valid_paths,
            "test": test_paths
        }
        
        # 移动文件并更新metadata
        set_metadata = {"train": [], "valid": [], "test": []}
        for set_type, paths in path_sets.items():
            for path in paths:
                idx = sample_paths.index(path)
                set_metadata[set_type].append(metadata_list[idx])
                
                # # 移动音频文件
                # base_name = os.path.basename(path)
                # for src, dst in [
                #     (os.path.join(path, "mixture.wav"), 
                #      os.path.join(datapath, set_type, "mixture", f"{base_name}.wav")),
                #     (os.path.join(path, "source1.wav"),
                #      os.path.join(datapath, set_type, "source1", f"{base_name}.wav")),
                #     (os.path.join(path, "source2.wav"),
                #      os.path.join(datapath, set_type, "source2", f"{base_name}.wav"))
                # ]:
                #     if os.path.exists(src):
                #         os.system(f"cp {src} {dst}")
        
        # 为每个集合创建CSV文件
        for set_type in ["train", "valid", "test"]:
            create_custom_dataset(
                datapath,
                savepath,
                set_metadata[set_type],
                dataset_name="HRSP",
                set_types=[set_type]
            )
    else:
        raise ValueError("Unknown dataset")


def create_custom_dataset(
    datapath,
    savepath,
    metadata_list,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture": "mixture",
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """
    os.makedirs(savepath, exist_ok=True)

    for set_type in set_types:
        # 直接从metadata获取音频路径
        mix_fl_paths = [meta["merged_audio_path"] for meta in metadata_list]
        s1_fl_paths = [meta["human_audio_path"] for meta in metadata_list]
        s2_fl_paths = [meta["robot_audio_path"] for meta in metadata_list]

        # 确保所有路径都存在
        for paths in [mix_fl_paths, s1_fl_paths, s2_fl_paths]:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Audio file not found: {path}")

        # 先定义基本的 CSV 列
        base_csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "human_wav",
            "human_wav_format",
            "human_wav_opts",
            "robot_wav",
            "robot_wav_format",
            "robot_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        # 获取 metadata 中的所有键
        all_metadata_keys = set()
        for metadata in metadata_list:
            all_metadata_keys.update(metadata.keys())

        # 移除可能重复的键
        if "background_noise_name" in all_metadata_keys:
            all_metadata_keys.remove("background_noise_name")

        # 合并列名
        csv_columns = base_csv_columns + list(all_metadata_keys)

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"),
            "w",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, human_path, robot_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):
                metadata = metadata_list[i]
                background_noise_name = metadata.get("background_noise_name", "")
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": os.path.abspath(mix_path),
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "human_wav": os.path.abspath(human_path),
                    "human_wav_format": "wav",
                    "human_wav_opts": None,
                    "robot_wav": os.path.abspath(robot_path),
                    "robot_wav_format": "wav",
                    "robot_wav_opts": None,
                    "noise_wav": background_noise_name,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                # 补全 metadata 中的其他字段
                for key in all_metadata_keys:
                    if key in metadata:
                        row[key] = metadata[key]
                writer.writerow(row)

if __name__ == "__main__":
    # print(os.getcwd())
    # print(os.path.exists(path=os.path.join(os.getcwd(), "data/HRSP2mix/raw/")))
    prepare_dataset(
        datapath=os.path.join(os.getcwd(), "data/HRSP2mix/raw/"),
        savepath=os.path.join(os.getcwd(), "data/HRSP2mix/processed/"),
        n_spks=2,
        skip_prep=False,
    )