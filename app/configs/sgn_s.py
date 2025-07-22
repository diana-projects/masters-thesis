import os
from .base import BaseEvalConfig

class SGNSConfig(BaseEvalConfig):
    def __init__(self):
        self.model_name = "SGN-S"
        self.cwd = "SGN" 
        self.program = f"./{self.cwd}/tools/test.py"
        self.config_path = f"./{self.cwd}/projects/configs/sgn/sgn-S-one-stage-guidance.py"
        self.ckpt_path = f"./{self.cwd}/ckpts/sgn-s-epoch_28.pth"
        self.eval_metric = "bbox"
        self.model_type="S"
        self.eval_mode = "--eval"
        self.extra_args = ["--deterministic"]

        self.pythonpath = os.environ.get("PYTHONPATH", "SGN")

    def get_args(self):
        return [
            self.config_path,
            self.ckpt_path,
            self.eval_mode,
            self.eval_metric,
            *self.extra_args
        ]
    
    def get_env(self):
        return {
            "PYTHONPATH": self.pythonpath,
            "MODEL_VERSION": self.model_name,
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
