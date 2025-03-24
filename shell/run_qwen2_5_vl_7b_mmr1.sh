set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

srun --partition $PARTITION \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=$QUOTA_TYPE \
    bash examples/run_qwen2_5_vl_7b_mmr1.sh