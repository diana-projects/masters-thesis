from pytorch_lightning import Trainer
from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule

import hydra
from omegaconf import DictConfig
import torch
import os
from hydra.utils import get_original_cwd

config_path= os.getenv('DATA_CONFIG')

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def calculate_parameter_number(model):
    total_params = sum(p.numel() for p in model.parameters())
    def format_params(n_params):
        if n_params >= 1e9:
            return f"{n_params / 1e9:.2f}B"
        elif n_params >= 1e6:
            return f"{n_params / 1e6:.2f}M"
        elif n_params >= 1e3:
            return f"{n_params / 1e3:.1f}K"
        else:
            return str(n_params)
    print(f"Number of parameters: {format_params(total_params)}")

def calculate_max_memory_allocated(model, kwargs):
    def format_memory(bytes_val):
        if bytes_val >= 1 << 40:
            return f"{bytes_val / (1 << 40):.2f} TB"
        elif bytes_val >= 1 << 30:
            return f"{bytes_val / (1 << 30):.2f} GB"
        elif bytes_val >= 1 << 20:
            return f"{bytes_val / (1 << 20):.2f} MB"
        elif bytes_val >= 1 << 10:
            return f"{bytes_val / (1 << 10):.2f} KB"
        else:
            return f"{bytes_val} B"
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        x=model(**kwargs)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory during inference: {format_memory(peak_memory)}")


def calculate_inference_time(model, kwargs):
    import time
    starttime = time.time()
    with torch.no_grad():
        x = model(**kwargs)
    print("Inference time: ", time.time() - starttime)

def eval_test(model,data_loader):
  for i, data in enumerate(data_loader):
        with torch.no_grad():
            pass

@hydra.main(config_name=config_path)
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    load_strict = True
    if config.dataset == "kitti":
        config.batch_size_per_gpu = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
        )

    elif config.dataset == "NYU":
        config.batch_size_per_gpu = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = NYUDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            use_depth_gt=config.use_depth_gt,
        )
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )

    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator="ddp"
    )

    model_path = os.path.join(get_original_cwd(), "OccDepth","trained_models", "occdepth.ckpt")

    print(
        "##### Max CUDA memory before load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )
    model = OccDepth.load_from_checkpoint(
        model_path,
        full_scene_size=full_scene_size,
        config=config,
        strict=load_strict,
    )
    model.cuda()
    model.eval()
    # calculate_model_size(model)
    # calculate_parameter_number(model)
    # calculate_inference_time(model, {"return_loss":False, "rescale":True, **data})
    # calculate_max_memory_allocated(model, {"return_loss":False, "rescale":True, **data})
    print(
        "##### Max CUDA memory after load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )

    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)
    print(
        "##### Max CUDA memory during all evaluation process: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )

if __name__ == "__main__":
    main()
