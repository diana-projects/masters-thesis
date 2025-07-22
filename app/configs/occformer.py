import os
from .base import BaseEvalConfig


class OccFormerConfig(BaseEvalConfig):
    def __init__(self):
        self.model_name = "OccFormer"
        self.cwd = self.model_name  # Assuming ${workspaceFolder} is the root
        self.program = f"./{self.cwd}/tools/test.py"
        self.config_path = f"./{self.cwd}/projects/configs/occformer_kitti/occformer_kitti.py"
        self.ckpt_path = f"./{self.cwd}/pretrain/occformer_kitti.pth"
        self.eval_mode = "--eval"
        self.eval_metric = "bbox"
        self.extra_args = ["--deterministic"]

        self.pythonpath = os.environ.get("PYTHONPATH", "OccFormer")

    def get_args(self):
        return [
            self.config_path,
            self.ckpt_path,
            f"{self.eval_mode}={self.eval_metric}"
        ]

    def get_env(self):
        return {
            "PYTHONPATH": self.pythonpath
    }

    def display_config(self):
        print(f"Model Name: {self.model_name}")
        print(f"Program Path: {self.program}")
        print(f"Config Path: {self.config_path}")
        print(f"Checkpoint Path: {self.ckpt_path}")
        print(f"Eval Mode: {self.eval_mode}")
        print(f"Eval Metric: {self.eval_metric}")
        print(f"Extra Args: {self.extra_args}")
        print(f"CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        print(f"PYTHONPATH: {self.pythonpath}")