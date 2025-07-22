import os
from .base import BaseEvalConfig

class CGFormerConfig(BaseEvalConfig):
    def __init__(self):
        self.model_name = "CGFormer"
        self.cwd = self.model_name  # ${workspaceFolder}
        self.program = f"./{self.cwd}/main.py"
        self.eval = True
        self.ckpt_path = f"./{self.cwd}/ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt"
        self.config_path = f"./{self.cwd}/configs/CGFormer-Efficient-Swin-SemanticKITTI.py"
        self.log_folder = f"./{self.cwd}/CGFormer-Efficient-Swin-SemanticKITTI-eval"
        self.seed = 7240
        self.log_every_n_steps = 100


    def get_args(self):
        args = []
        if self.eval:
            args.append("--eval")
        args.extend([
            "--ckpt_path", self.ckpt_path,
            "--config_path", self.config_path,
            "--log_folder", self.log_folder,
            "--seed", str(self.seed),
            "--log_every_n_steps", str(self.log_every_n_steps),
        ])
        return args
    
    def get_env(self):
        return {
        }

    def display_config(self):
        print(f"Model Name: {self.model_name}")
        print(f"Program Path: {self.program}")
        print(f"Checkpoint Path: {self.ckpt_path}")
        print(f"Config Path: {self.config_path}")
        print(f"Log Folder: {self.log_folder}")
        print(f"Seed: {self.seed}")
        print(f"Log Every N Steps: {self.log_every_n_steps}")
        print(f"CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        print(f"Eval Mode: {self.eval}")
