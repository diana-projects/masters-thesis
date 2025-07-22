# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os.path as osp
import shutil
import tempfile
import time
from pathlib import Path
from typing import Union

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from projects.mmdet3d_plugin.sgn.utils.ssc_metric import SSCMetrics
import numpy as np

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return '{:.3f}MB'.format(size_all_mb)

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
    return format_params(total_params)

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

    return format_memory(peak_memory)

def calculate_inference_time(model, kwargs):
    starttime = time.time()
    with torch.no_grad():
        x = model(**kwargs)
    print("Inference time: ", time.time() - starttime)

    return str(time.time()-starttime)


def custom_single_gpu_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    save_results: bool = True,
    save_path: Union[Path, str] = None,
):
    model.eval()

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    # evaluate ssc
    ssc_metric = SSCMetrics(len(dataset.class_names)).cuda()
    if save_results:
        import os
        save_path = os.environ['RESULTS_PATH'] if os.environ['RESULTS_PATH'] else Path("./results")
        save_path = os.path.join(save_path, os.environ['MODEL_VERSION'])
        os.makedirs(save_path, exist_ok=True)
        save_path = Path(save_path)
    technical_dict = {}
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if i==0:
                model_size = calculate_model_size(model)
                param_number = calculate_parameter_number(model)
                inference_time = calculate_inference_time(model, {"return_loss":False, "rescale":True, **data})
                max_mem_allocated = calculate_max_memory_allocated(model, {"return_loss":False, "rescale":True, **data})
                technical_dict.update({
                    "model_size": model_size,
                    "param_number": param_number,
                    "inference_time": inference_time,
                    "max_mem_allocated": max_mem_allocated,
                })
        output_voxels = torch.argmax(result['output_voxels'], dim=1)
        target_voxels = result['target_voxels'].clone()
        ssc_metric.update(
            y_pred=output_voxels,
            y_true=target_voxels,
        )
        if save_results:
            save_filename = (
                Path(
                    data["img_metas"]._data[0][0][0]["frame_id"],
                )
                .with_suffix(".npy")
                .name
            )
            np.save(save_path/ save_filename, output_voxels.cpu())
        batch_size = output_voxels.shape[0]
        for _ in range(batch_size):
            prog_bar.update()

    res = {
        'ssc_scores': ssc_metric.compute(),
    }
    import pickle 
    cleaned = {
        k: (
            # detach+move to CPU once
            v.detach().cpu().item()
            if isinstance(v, torch.Tensor) and v.numel() == 1
            else (v.tolist() if isinstance(v, torch.Tensor) else v)
        )
        for k, v in res['ssc_scores'].items()
        }
    with open(os.path.join(save_path,'inference_result.p'), 'wb') as fp:
        pickle.dump(cleaned, fp)
    with open(os.path.join(save_path,'technical_details.p'), 'wb') as fp:
        pickle.dump(technical_dict, fp)
    return res


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    # evaluate ssc
    ssc_metric = SSCMetrics(len(dataset.class_names)).cuda()
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
        if result['target_voxels'] is not None:
            target_voxels = result['target_voxels'].clone()
            ssc_results_i = ssc_metric.compute_single(
                y_pred=output_voxels, y_true=target_voxels)
            ssc_results.append(ssc_results_i)
            
        batch_size = output_voxels.shape[0]
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    return res


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results
