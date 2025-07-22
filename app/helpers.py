import numpy as np
import pickle

def load_npy(file: str):
   return np.load(file)[0]

def load_label(file: str):
    VOXEL_DIMS = (256, 256, 32)
    scan_data = np.fromfile(
        file,
        dtype=np.uint16,
    ).reshape(VOXEL_DIMS)
    return scan_data

def load_pkl(file: str):
   try:
      loaded_obj = pickle.load(file) 
   except Exception as e:
      print("File does not have read or readlines attributes, opening it in bytes mode.")
      with open(file,"rb") as f:
         loaded_obj = pickle.load(f)
   return loaded_obj["y_pred"]

loading_map = {
    "cgformer": ".label",
    "htcl": ".npy",
    "occdepth": ".pkl",
    "occformer": ".npy",
    "stereoscene": ".npy",
    "sgn-t": ".npy",
    "sgn-l": ".npy",
    "sgn-s": ".npy",
}

loading_map_func = {
    ".pkl": load_pkl,
    ".npy": load_npy,
    ".label": load_label,
}

model_mapper = {
    "htcl": "HTCL",
    "stereoscene": "StereoScene",
    "sgn-l": "SGN-L",
    "sgn-t": "SGN-T",
    "sgn-s": "SGN-S",
    "occformer": "OccFormer",
    "occdepth": "OccDepth",
    "cgformer": "CGFormer",
}

def compute_iou_(prediction: np.ndarray, ground_truth: np.ndarray, num_classes: int, ignore_label: int = 0):
    """
    Compute per-class IoU, mean IoU, and overall IoU for semantic scene completion.

    Parameters:
    -----------
    prediction : np.ndarray
        The predicted 3D voxel grid (same shape as ground_truth), with class labels.
    ground_truth : np.ndarray
        The ground truth 3D voxel grid, with class labels.
    num_classes : int
        Total number of semantic classes (including ignored).
    ignore_label : int
        Label to be ignored when calculating IoU (commonly 0 or 255).

    Returns:
    --------
    per_class_iou : dict
        Dictionary mapping class index to its IoU.
    mean_iou : float
        Mean IoU over classes that are present in the ground truth.
    overall_iou : float
        IoU across all non-ignored classes combined.
    """

    assert prediction.shape == ground_truth.shape, "Prediction and GT must have same shape."

    per_class_iou = [0 for _ in range(num_classes)] 
    intersection_total = 0
    union_total = 0

    for class_id in range(num_classes):
        if class_id == ignore_label:
            continue

        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union > 0:
            iou = intersection / union
            per_class_iou[class_id] = iou
            intersection_total += intersection
            union_total += union
        else:
            # Class not present in ground truth or prediction, skip
            continue

    mean_iou = np.mean(list(per_class_iou)) if per_class_iou else 0.0
    overall_iou = intersection_total / union_total if union_total > 0 else 0.0

    return per_class_iou, mean_iou, overall_iou

def load_by_model(file:str, model_name:str):
    if model_name not in loading_map:
        raise ValueError(
            f"We do not have {model_name} available, please choose among {','.join(loading_map.keys())}",
        )
    load_func = loading_map_func[loading_map[model_name]]
    project_result = load_func(file)
    return project_result
