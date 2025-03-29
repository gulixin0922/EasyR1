set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/train-00000-of-00001.parquet \
    data.val_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/validation-00000-of-00001.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=6144 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_mmr1 \
    trainer.n_gpus_per_node=8 \
    trainer.val_freq=10 \
    trainer.save_freq=10 \
    trainer.load_checkpoint_path=/mnt/petrelfs/gulixin/workspace/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_mmr1/global_step_20
