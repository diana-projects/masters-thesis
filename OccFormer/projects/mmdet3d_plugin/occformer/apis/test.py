# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from fvcore.nn import parameter_count_table
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from projects.mmdet3d_plugin.utils import SSCMetrics, cm_to_ious, format_results

# utils for saving predictions
from .utils import *

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
    show_score_thr=0.3,
    pred_save=None,
    test_save=None,
    save_results: bool = True,
    save_path=None,
):
    model.eval()

    if save_results:
        save_path = os.environ['RESULTS_PATH'] if os.environ['RESULTS_PATH'] else Path("./results")
        save_path = os.path.join(save_path, "OccFormer")
        os.makedirs(save_path, exist_ok=True)
        save_path = Path(save_path)

    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()

    # evaluate lidarseg
    evaluation_semantic = 0

    # evaluate ssc
    is_semkitti = hasattr(dataset, 'camera_used')
    ssc_metric = SSCMetrics().cuda()
    logger.info(parameter_count_table(model, max_depth=4))

    batch_size = 1

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
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']

            # for one-gpu test, print results for each batch
            ious = cm_to_ious(evaluation_semantic)
            res_table, _ = format_results(ious, return_dic=True)
            print(res_table)

        img_metas = data['img_metas'].data[0][0]
        # save for test submission
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            target_voxels = result['target_voxels'].clone()
            ssc_metric.update(y_pred=output_voxels,  y_true=target_voxels)

            if save_results:
                save_filename = Path(data['img_metas'].data[0][0]["frame_id"]).with_suffix(".npy").name
                np.save(save_path / save_filename, output_voxels.cpu())
            # compute metrics
            scores = ssc_metric.compute()
            if is_semkitti:
                print('\n Evaluating semanticKITTI occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            else:
                print('\n Evaluating nuScenes occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))

            # save for val predictions, mostly for visualization
            if pred_save is not None:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)

                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'], 
                        target_points=result['target_points'], 
                        save_path=pred_save, 
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))

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
    if type(evaluation_semantic) is np.ndarray:
        res['evaluation_semantic'] = evaluation_semantic
    with open(os.path.join(save_path,'technical_details.p'), 'wb') as fp:
        pickle.dump(technical_dict, fp)
    return res


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, pred_save=None, test_save=None):
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
    ssc_metric = SSCMetrics().cuda()
    is_semkitti = hasattr(dataset, 'camera_used')
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    is_val_save_predictins = pred_save is not None
    if is_val_save_predictins:
        os.makedirs(pred_save, exist_ok=True)
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
        
        img_metas = data['img_metas'].data[0][0]
        # occupancy prediction
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
            if result['target_voxels'] is not None:
                target_voxels = result['target_voxels'].clone()
                ssc_results_i = ssc_metric.compute_single(
                    y_pred=output_voxels, y_true=target_voxels)
                ssc_results.append(ssc_results_i)
            
            if is_val_save_predictins:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'],
                        target_points=result['target_points'], 
                        save_path=pred_save,
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    
    if is_test_submission:
        return None
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    if type(evaluation_semantic) is np.ndarray:
        # convert to tensor for reduce_sum
        evaluation_semantic = torch.from_numpy(evaluation_semantic).cuda()
        dist.all_reduce(evaluation_semantic, op=dist.ReduceOp.SUM)
        res['evaluation_semantic'] = evaluation_semantic.cpu().numpy()
    
    return res
