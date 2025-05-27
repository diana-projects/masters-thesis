from pytorch_lightning import Trainer
from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle
from ssc_metric import SSCMetrics

def to_cuda(datas):
    assert isinstance(datas, list)
    for i, data in enumerate(datas):
        datas[i] = data.cuda()

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

def calculate_max_memory_allocated(model, batch):
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
        x=model(batch)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory during inference: {format_memory(peak_memory)}")


def calculate_inference_time(model, batch):
    import time
    starttime = time.time()
    with torch.no_grad():
        x = model(batch)
    print("Inference time: ", time.time() - starttime)


@hydra.main(config_name=config_path)
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    load_strict = True

    # Setup dataloader
    if config.dataset == "kitti":
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
        data_module.setup()
        data_loader = data_module.val_dataloader()

    elif config.dataset == "NYU":
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
        data_module.setup()
        data_loader = data_module.val_dataloader()
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
    else:
        print("dataset not support")

    # Load pretrained models
    model_path = os.path.join(get_original_cwd(), "trained_models", "occdepth.ckpt")

    model = OccDepth.load_from_checkpoint(
        model_path,
        full_scene_size=full_scene_size,
        config=config,
        strict=load_strict,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config_path,"../../../../output", config.dataset)
    output_path = os.path.abspath(output_path)
    metrics = SSCMetrics()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            batch["img"] = batch["img"].cuda()
            to_cuda(batch["T_velo_2_cam"])
            to_cuda(batch["cam_k"])
            to_cuda(batch["ida_mats"])

            pred = model(batch)
            # calculate_model_size(model)
            # calculate_parameter_number(model)
            # calculate_inference_time(model, batch)
            # calculate_max_memory_allocated(model, batch)
            
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            # if 'target' in batch:
            #     y_true = batch["target"].to(y_pred.device)
            #     metrics.update(y_pred, y_true)
            for i in range(config.batch_size_per_gpu):
                out_dict = {"y_pred": np.argmax(pred['ssc_logit'].detach().cpu().numpy(), axis=1)[i]}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )
                    metrics.update(torch.tensor(out_dict['y_pred'].astype(np.int16)), torch.tensor(out_dict['target'].astype(np.int16)),)
            
                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                elif config.dataset == "tartanair":
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["vox_origin"] = np.array([-6, -3, 0])  # cam coord
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                elif config.dataset == "kitti":
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )
                # import pdb; pdb.set_trace()

                if idx%5==0:
                    os.makedirs(write_path, exist_ok=True)
                    with open(filepath, "wb") as handle:
                        pickle.dump(out_dict, handle)
                        print("wrote to", filepath)

        print("SSC metrics:", metrics.compute())
if __name__ == "__main__":
    main()
