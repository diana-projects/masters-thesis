import subprocess
import os
import threading
from app.configs.cgformer import CGFormerConfig
from app.configs.htcl import HTCLConfig
from app.configs.occdepth import OccDepthConfig
from app.configs.occformer import OccFormerConfig
from app.configs.sgn_l import SGNLConfig
from app.configs.sgn_s import SGNSConfig
from app.configs.sgn_t import SGNTConfig
from app.configs.stereoscene import StereoSceneConfig

MODEL_CONFIGS = {
    "cgformer": CGFormerConfig,
    "sgn-t": SGNTConfig,
    "sgn-s": SGNSConfig,
    "sgn-l": SGNLConfig,
    "occformer": OccFormerConfig,
    "occdepth": OccDepthConfig,
    "stereoscene": StereoSceneConfig,
    "htcl": HTCLConfig,
}

ENV_PATH = {
    "occdepth": "occdepth_torch_1_10",
    "stereoscene": "stereoscene3.7", 
    "htcl": "stereoscene3.7", 
    "sgn-t": "SGN_3.8",
    "sgn-s": "SGN_3.8",
    "sgn-l": "SGN_3.8",
}

def run_model_in_env(env_name: str, cuda_device: int):
    if env_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown environment name: {env_name}")

    config = MODEL_CONFIGS[env_name]() 
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.update(config.get_env())
    env_path = ENV_PATH.get(env_name, env_name) 

    PYTHON_PATH = f"/dgx_data1/aclx58n/envs/{env_path}/bin/python"

    process = subprocess.Popen(
        [PYTHON_PATH, config.program, *config.get_args()],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def stream_output(pipe, name):
        for line in iter(pipe.readline, ''):
            print(f"[{name}] {line.strip()}")

    threading.Thread(
        target=stream_output,
        args=(process.stdout, "STDOUT"),
        daemon=True,
    ).start()

    threading.Thread(
        target=stream_output,
        args=(process.stderr, "STDERR"),
        daemon=True,
    ).start()
    process.wait()
