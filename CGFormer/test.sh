CUDA_VISIBLE_DEVICES=3 python main.py \
--eval \
--ckpt_path ./logs/CGFormer-Efficient-Swin-SemanticKITTI.ckpt \
--config_path configs/CGFormer-Efficient-Swin-SemanticKITTI.py \
--log_folder version1 \
--seed 7240 \
--log_every_n_steps 100 \
> CGFormer-Efficient-Swin-SemanticKITTI-Eval.log 2>&1 &