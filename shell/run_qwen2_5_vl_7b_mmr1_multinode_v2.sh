#!/bin/bash

#SBATCH --partition=Intern5
#SBATCH --job-name=qwen2_5_mmr1_node4
#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --quotatype=reserved
#SBATCH --mem=200G
#SBATCH --output=logs/qwenvl_v2_5/grpo/%x-%j.log
#SBATCH --error=logs/qwenvl_v2_5/grpo/%x-%j.err

export RAY_TMPDIR=/dev/shm/ray_glx
rm -rf ${RAY_TMPDIR}
mkdir -p ${RAY_TMPDIR}

CURR_DIR="/mnt/petrelfs/gulixin/workspace/EasyR1"
JOBLOG="${CURR_DIR}/logs/qwenvl_v2_5/grpo/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"
OUTPUT_DIR="${CURR_DIR}/outputs/qwenvl_v2_5/grpo/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
rm -rf ${JOBLOG}
mkdir -p $(dirname "$JOBLOG")
mkdir -p ${OUTPUT_DIR}

echo "pwd=$(pwd)" >> ${JOBLOG}
echo "JOBLOG=${JOBLOG}" >> ${JOBLOG}
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." >> ${JOBLOG}
echo "The job name is: ${SLURM_JOB_NAME}" >> ${JOBLOG}
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}" >> ${JOBLOG}
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}" >> ${JOBLOG}
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}" >> ${JOBLOG}
echo "SLURM_NNODES: ${SLURM_NNODES}" >> ${JOBLOG}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head" &>> ${JOBLOG}

# make sure we set environment variables before Ray initialization
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

echo "Starting HEAD at $head_node" &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --dashboard-host=0.0.0.0 --dashboard-port=8265 \
        --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 8 --block &>> ${JOBLOG} &
sleep 20

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "worker_num: $worker_num" &>> ${JOBLOG}
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i" &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 8 --block &>> ${JOBLOG} &
    sleep 10
done
sleep 20

# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
MODEL_PATH=/mnt/petrelfs/gulixin/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/68156fd997cdc9f710620466735af49862bb81f6
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

echo "submit ray job" &>> ${JOBLOG}
# srun --overlap --nodes=1 --ntasks=1 --gres=gpu:0 -w "$head_node" \
#     python3 -m verl.trainer.main \
#         config=examples/grpo_example.yaml \
#         data.train_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/train-00000-of-00001.parquet \
#         data.val_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/validation-00000-of-00001.parquet \
#         data.system_prompt="${SYSTEM_PROMPT}" \
#         data.max_prompt_length=6144 \
#         worker.actor.model.model_path=${MODEL_PATH} \
#         worker.rollout.enable_chunked_prefill=false \
#         trainer.experiment_name=qwen2_5_vl_7b_mmr1_multinode \
#         trainer.n_gpus_per_node=8 \
#         trainer.nnodes=${SLURM_NNODES} \
#         trainer.save_checkpoint_path=${OUTPUT_DIR} \
#         trainer.val_freq=10 \
#         trainer.save_freq=10 &>> ${JOBLOG}

# 需要关闭代理proxy_off，否则连不上http://localhost:8265
srun --overlap --nodes=1 --ntasks=1 --gres=gpu:0 -w "$head_node" \
    ray job submit --address=http://localhost:8265 \
        -- python3 -m verl.trainer.main \
        config=examples/grpo_example.yaml \
        data.train_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/train-00000-of-00001.parquet \
        data.val_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/validation-00000-of-00001.parquet \
        data.system_prompt="${SYSTEM_PROMPT}" \
        data.max_prompt_length=6144 \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.rollout.enable_chunked_prefill=false \
        trainer.experiment_name=qwen2_5_vl_7b_mmr1_multinode \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_checkpoint_path=${OUTPUT_DIR} \
        trainer.logger=['console'] \
        trainer.val_freq=10 \
        trainer.save_freq=10 &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}