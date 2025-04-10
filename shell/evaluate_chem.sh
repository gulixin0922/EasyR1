set -x

CHECKPOINT=${1}

export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

BASENAME=$(basename "${CHECKPOINT}")

if [[ "${BASENAME}" =~ checkpoint-[0-9]+ ]]; then
  OUT_PATH="results/$(basename "$(dirname "${CHECKPOINT}")")/$(basename "${CHECKPOINT}")"
else
  OUT_PATH="results/${BASENAME}"
fi

echo "OUT_PATH: ${OUT_PATH}"

if [[ ! -d "${OUT_PATH}" ]]; then
  mkdir -p "${OUT_PATH}"
fi

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

export OPENAI_API_KEY=''
export OPENAI_PROXY_URL=''
export OPENAI_BASE_URL=''

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"


# DATASETS=("CMMU-Base")
# DATASETS=("CMMU-Thinking")
# DATASETS=("mmcr_post-Base")
DATASETS=("mmcr_post-Thinking")
for dataset in "${DATASETS[@]}"; do
  # echo "submit: $dataset"
  torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  eval/chemvlm/evaluate.py --checkpoint ${CHECKPOINT} --datasets $dataset --out-dir ${OUT_PATH} \
  2>&1 | tee -a "${OUT_PATH}/log_infer.txt"

  if [[ ${dataset} == "CMMU-Base" || ${dataset} == "CMMU-Thinking" ]]; then
    python -u  eval/chemvlm/test_exam_performance.py --out-dir ${OUT_PATH} --datasets $dataset 2>&1 | tee -a "${OUT_PATH}/log_score.txt"
  elif [[ ${dataset} == "mmcr_post-Base" || ${dataset} == "mmcr_post-Thinking" ]]; then
    python -u  eval/chemvlm/test_exam_performance.py --out-dir ${OUT_PATH} --datasets $dataset 2>&1 | tee -a "${OUT_PATH}/log_score.txt"
  fi

done
