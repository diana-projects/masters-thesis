from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from app.run_model import run_model_in_env
import os
import uvicorn
import dotenv
import numpy as np
from pathlib import Path
import logging
import pickle
from app.helpers import loading_map, load_by_model,load_label, model_mapper, compute_iou_
import yaml 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inference API with Models and 3D",
    description="API for running inference with multiple models and 3D visualization",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8008",
        "http://127.0.0.1:8008",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Available models
AVAILABLE_MODELS = ["htcl", "cgformer", "sgn-s", "sgn-t", "sgn-l", "occformer", "occdepth", "stereoscene"]

# Pydantic models
class Sample(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

class InferenceRequest(BaseModel):
    sample_id: str
    model: str

class InferenceResult(BaseModel):
    sample_id: str
    model: str
    result: dict
    timestamp: str
    status: str

class Config(BaseModel):
    image_folder_path: str
    dataset_path: str
    available_models: List[str]

def get_samples_list():
    """Your existing function to get samples from dataset"""
    try:
        dotenv.load_dotenv("../.env")
        dataset_path = os.environ.get('DATASET_PATH')
        if not dataset_path:
            raise RuntimeError("DATASET_PATH not set in .env")
        
        label_dir = os.path.join(dataset_path, "dataset", "sequences", "08", "voxels")
        if not os.path.exists(label_dir):
            raise RuntimeError(f"Label directory not found: {label_dir}")
        
        ids = list(Path(label_dir).rglob("*.label"))
        logger.info(f"Found {len(ids)} files in {label_dir}")
        
        samples = []
        for id_ in ids:
            sample_id = id_.stem
            samples.append({
                    "id": sample_id,
                    "name": f"Sample file {int(sample_id)}",
                })
             
        logger.info(f"Processed {len(samples)} valid samples")
        return samples
        
    except Exception as e:
        logger.error(f"Error in get_samples_list: {e}")
        raise

def get_dataset_info():
    """Get dataset path and image directory info"""
    dotenv.load_dotenv("../.env")
    dataset_path = os.environ.get('DATASET_PATH')
    if not dataset_path:
        raise RuntimeError("DATASET_PATH not set in .env")
    
    image_dir = os.path.join(dataset_path, "dataset", "sequences", "08", "image_2")
    if not os.path.exists(image_dir):
        raise RuntimeError(f"Image directory not found: {image_dir}")

    results_path = os.environ.get('RESULTS_PATH') 
    if not results_path:
        raise RuntimeError("RESULTS_PATH not set in .env")
    
    return {
        "dataset_path": dataset_path,
        "image_dir": image_dir,
        "results_path": results_path
    }

def get_labels_path():
    dataset_info = get_dataset_info()
    result_dir = os.path.join(dataset_info['dataset_path'],  "dataset", "sequences","08","voxels")
    return result_dir

def get_result_path(model: str):
    """Get the path to result files for a given sample and model"""
    dataset_info = get_dataset_info()
    result_dir = os.path.join(dataset_info['results_path'], model_mapper.get(model, model))
    return result_dir

@app.get("/")
async def root():
    return {"message": "Inference API with Models and 3D is running", "status": "healthy"}

@app.get("/config")
async def get_config():
    """Get configuration including dataset paths and available models"""
    try:
        dataset_info = get_dataset_info()
        config = {
            "image_folder_path": dataset_info["image_dir"],
            "dataset_path": dataset_info["dataset_path"],
            "available_models": AVAILABLE_MODELS
        }
        
        # Add no-cache headers
        response = JSONResponse(content=config)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
        
    except RuntimeError as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/samples")
async def get_samples():
    """Get list of available sample files from your dataset"""
    try:
        logger.info("Fetching samples...")
        samples_data = get_samples_list()
        
        samples = [
            {
                "id": sample["id"],
                "name": sample["name"],
                "description": f"Dataset sequence 08, image {sample['id']}"
            }
            for sample in samples_data
        ]
        
        logger.info(f"Returning {len(samples)} samples")
        
        # Add no-cache headers
        response = JSONResponse(content=samples)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
        
    except RuntimeError as e:
        logger.error(f"Samples error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error loading samples: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading samples: {str(e)}")

@app.get("/image/{sample_id}")
async def get_image(sample_id: str):
    """Serve image files from the dataset"""
    try:
        dataset_info = get_dataset_info()
        image_dir = dataset_info["image_dir"]
        
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        for ext in extensions:
            file_path = Path(image_dir) / f"{sample_id}{ext}"
            if file_path.exists() and file_path.is_file():
                return FileResponse(
                    path=file_path,
                    media_type=f"image/{ext[1:]}",
                    filename=f"{sample_id}{ext}",
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Pragma": "no-cache",
                        "Expires": "0"
                    }
                )
        
        raise HTTPException(status_code=404, detail=f"Image not found for sample {sample_id}")
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.get("/scene/{sample_id}")
async def get_scene(sample_id: str, model: str = Query(...)):
    """Serve 3D scene data (.npy files) for visualization"""
    try:
        if model not in AVAILABLE_MODELS and model!='ground_truth':
            raise HTTPException(status_code=400, detail=f"Invalid model. Available: {AVAILABLE_MODELS}")
        elif model=='ground_truth':
            labels_dir = get_labels_path()
            file_path = Path(labels_dir) / f"{sample_id}.label" 
            data = load_label(file_path).transpose(2,0,1)
            return Response(
                content=data.astype(np.float32).tobytes(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={file_path}",
                    "X-Array-Shape": str(data.shape),
                    "X-Array-Dtype": str(data.dtype),
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )

        result_dir = get_result_path(model)
        file_path = Path(result_dir) / f"{sample_id}{loading_map[model]}" 
        if file_path.exists() and file_path.is_file():
            data = load_by_model(file_path, model).transpose(2,0,1)
            return Response(
                content=data.astype(np.float32).tobytes(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={file_path}",
                    "X-Array-Shape": str(data.shape),
                    "X-Array-Dtype": str(data.dtype),
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )

        raise HTTPException(status_code=404, detail=f"3D scene data not found for sample {sample_id} with model {model}")

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving 3D scene: {str(e)}")

from fastapi import status
@app.post("/run_batch_inference/{model}")
async def run_batch_inference(model: str):

    dotenv.load_dotenv("../.env")
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model. Available: {sorted(AVAILABLE_MODELS)}"
        )

    try:
        run_model_in_env(model, cuda_device=2)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running inference: {e}"
        )

    numerical_results_path = os.path.join(os.environ['RESULTS_PATH'], model_mapper.get(model, model), "inference_result.p")
    with open(numerical_results_path, 'rb') as f:
        numerical_results = pickle.load(f)

    numerical_results["iou_ssc"] = (
        numerical_results["iou_ssc"]
        if isinstance(
            numerical_results["iou_ssc"],
            list, 
        )
        else numerical_results["iou_ssc"].tolist()
    )

    with open(os.path.join(os.environ['RESULTS_PATH'], model_mapper.get(model, model),'technical_details.p'), 'rb') as f:
        technical_dict = pickle.load(f)

    response_dict = {
        "status": "success",
        "total_samples": 815,
        "ssc_scores": numerical_results,
        "technical_dict": technical_dict
    }
    return response_dict

@app.post("/compute_iou/{sample_id}")
async def compute_iou(sample_id: str, model=Query(...)):
    """Run inference on selected sample with specified model"""
    try:
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model. Available: {AVAILABLE_MODELS}")

        labels_dir = get_labels_path()
        gt_file = Path(labels_dir) / f"{sample_id}.label"
        if not gt_file.exists():
            raise HTTPException(404, detail="Ground-truth label not found")
        gt = load_label(gt_file)

        result_dir = get_result_path(model)
        pred_file = Path(result_dir) / f"{sample_id}{loading_map[model]}"
        if not pred_file.exists():
            raise HTTPException(404, detail=f"Predicted scene for model {model} not found")
        pred = load_by_model(pred_file, model)

        with open('./resources/semantic_kitti.yaml', 'r') as f:
            sem = yaml.safe_load(f)
        remapped_gt = np.array([sem['learning_map'][index] for index in gt.flatten()]).reshape(gt.shape)

       # assume compute_iou_ now returns (per_class_iou, mean_iou, overall_iou)
        per_class_iou, mean_iou, overall_iou = compute_iou_(pred, remapped_gt, num_classes=20)
        per_class_iou = [float(num) for num in per_class_iou]

        return {
            "sample_id": sample_id,
            "model": model,
            "per_class_iou": per_class_iou,
            "mean_iou": float(mean_iou),
            "iou": float(overall_iou),
            "status": "success"
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")

@app.get("/load_result")
async def load_result(sample_id: str, model: str = Query(...)):
    """Load existing result for sample with specified model"""
    try:
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model. Available: {AVAILABLE_MODELS}")
        
        model_folder = get_result_path(model)
        if not Path(model_folder).is_dir():
            raise HTTPException(
                status_code=404,
                detail="Currently there are no saved results for that model. Please firstly run batch inference"
            )
        
        samples = get_samples_list()
        sample_ids = [s["id"] for s in samples]
        
        if sample_id not in sample_ids:
            raise HTTPException(status_code=404, detail="Sample not found")

        result = {
            "cached_at": "2024-01-01T00:00:00Z",
            "model_version": "v1.1.0",
            "sequence": "08",
            "frame_id": sample_id,
            "model_used": model
        }
        
        return {
            "sample_id": sample_id,
            "model": model,
            "result": result,
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "success"
        }
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading result: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        dataset_info = get_dataset_info()
        samples_count = len(get_samples_list())
        return {
            "status": "healthy",
            "dataset_path": dataset_info["dataset_path"],
            "image_directory": dataset_info["image_dir"],
            "samples_count": samples_count,
            "available_models": AVAILABLE_MODELS
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    try:
        dataset_info = get_dataset_info()
        samples = get_samples_list()
        print(f"Dataset path: {dataset_info['dataset_path']}")
        print(f"Image directory: {dataset_info['image_dir']}")
        print(f"Found {len(samples)} samples")
        print(f"Available models: {AVAILABLE_MODELS}")
        
        uvicorn.run(
            "fastapi_fixed:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Please check your .env file and DATASET_PATH configuration")
