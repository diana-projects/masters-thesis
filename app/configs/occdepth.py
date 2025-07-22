import os
from .base import BaseEvalConfig

class OccDepthConfig(BaseEvalConfig):
    def __init__(self):
        self.model_name = "OccDepth"
        self.cwd = "OccDepth"  
        self.program = f"./{self.cwd}/occdepth/scripts/generate_output.py"

        # CLI arguments
        self.n_gpus = 1
        self.batch_size_per_gpu = 1

        # Environment variables
        self.pythonpath = os.environ.get("PYTHONPATH", f"{self.cwd}:$PYTHONPATH")
        self.data_log = os.environ.get("DATA_LOG", f"./{self.cwd}/logdir/b3_F32_FixMultiview")
        self.data_config = os.environ.get("DATA_CONFIG", f"../config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml")
        self.ets_toolkit = os.environ.get("ETS_TOOLKIT", "qt4")
        self.qt_api = os.environ.get("QT_API", "pyqt5")

    def get_args(self):
        return [
            f"n_gpus={self.n_gpus}",
            f"batch_size_per_gpu={self.batch_size_per_gpu}"
        ]
    
    def get_env(self):
        return {
            "PYTHONPATH": self.pythonpath,
            "DATA_LOG": self.data_log,
            "DATA_CONFIG": self.data_config,
            "ETS_TOOLKIT": self.ets_toolkit,
            "QT_API": self.qt_api
        }

    def display_config(self):
        print(f"Model Name: {self.model_name}")
        print(f"Program Path: {self.program}")
        print(f"n_gpus: {self.n_gpus}")
        print(f"batch_size_per_gpu: {self.batch_size_per_gpu}")
        print("Environment Variables:")
        print(f"  CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        print(f"  PYTHONPATH: {self.pythonpath}")
        print(f"  DATA_LOG: {self.data_log}")
        print(f"  DATA_CONFIG: {self.data_config}")
        print(f"  ETS_TOOLKIT: {self.ets_toolkit}")
        print(f"  QT_API: {self.qt_api}")