#!/bin/bash
set -x

# Getting the node names
PARTITION=Intern5
SLURM_CPUS_PER_TASK=10
SLURM_GPUS_PER_NODE=8
nodes_array=(HOST-10-140-60-21 HOST-10-140-66-196)
node_num=${#nodes_array[@]}
head_node=${nodes_array[0]}
head_node_ip=$(srun --partition $PARTITION --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

printenv

# echo "Starting HEAD at $head_node"
# # srun --partition $PARTITION --nodes=1 --ntasks=1 -w "$head_node" ray stop
# srun --partition $PARTITION --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gres=gpu:${SLURM_GPUS_PER_NODE} -w "$head_node" \
#     ray start --head --node-ip-address="$head_node_ip" --port=$port --dashboard-host=0.0.0.0 --dashboard-port=8265 \
#         --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --temp-dir /mnt/petrelfs/gulixin/ray_temp_multinode --block &
# sleep 20

# # number of nodes other than the head node
# worker_num=$(($node_num - 1))
# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     # srun --partition $PARTITION --nodes=1 --ntasks=1 -w "$node_i" ray stop
#     srun --partition $PARTITION --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gres=gpu:${SLURM_GPUS_PER_NODE} -w "$node_i" \
#         ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
#     sleep 10
# done
# sleep 20

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

srun --partition $PARTITION --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main \
        config=examples/grpo_example.yaml \
        data.train_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/train-00000-of-00001.parquet \
        data.val_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/validation-00000-of-00001.parquet \
        data.system_prompt="${SYSTEM_PROMPT}" \
        data.max_prompt_length=6144 \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.rollout.enable_chunked_prefill=false \
        trainer.experiment_name=qwen2_5_vl_7b_mmr1_multinode \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.nnodes=${node_num} \
        trainer.val_freq=10 \
        trainer.save_freq=10 2>&1 | tee verl_demo_slurm.log

# srun --partition $PARTITION --overlap --nodes=1 --ntasks=1 -w "$head_node" \
#     ray job submit --address="http://10.140.66.172:8265" \
#         -- python3 -m verl.trainer.main \
#             config=examples/grpo_example.yaml \
#             data.train_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/train-00000-of-00001.parquet \
#             data.val_files=/mnt/petrelfs/share_data/gulixin/r1_like_data/MMR1-Math-RL-Data-v0/data/validation-00000-of-00001.parquet \
#             data.system_prompt="${SYSTEM_PROMPT}" \
#             data.max_prompt_length=6144 \
#             worker.actor.model.model_path=${MODEL_PATH} \
#             worker.rollout.enable_chunked_prefill=false \
#             trainer.experiment_name=qwen2_5_vl_7b_mmr1_multinode \
#             trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
#             trainer.nnodes=${node_num} \
#             trainer.val_freq=10 \
#             trainer.save_freq=10 2>&1 | tee verl_demo_slurm.log
