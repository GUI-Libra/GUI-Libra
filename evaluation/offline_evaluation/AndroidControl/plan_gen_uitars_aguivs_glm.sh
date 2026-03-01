#!/bin/bash
set -e

############################
# Global config
############################
dir="path/to/your/dir"

# model_path="xlangai/Aguvis-7B-720P"
# exp_name="Aguvis-7B-720P"
# agent_family="aguvis"

model_path="zai-org/GLM-4.1V-9B-Thinking"
exp_name="GLM-4.1V-9B-Thinking_fix"
agent_family="glm"

# model_path="ByteDance-Seed/UI-TARS-1.5-7B"
# exp_name="UI-TARS-1.5-7B"
# agent_family="uitars"

add_template=1
reasoning=0

ports=(8000 8001 8002 8003 8004 8005 8006 8007)

input_file="data/500_steps_filtered_bbox_filtered_bbox_matchinstruction.json"
screenshot_dir="path/to/AndroidControl_images"

############################
# Run
############################
for level in high low; do

  echo "=============================="
  echo "LEVEL = ${level}"
  echo "=============================="

  ############################
  # temperature = 0.0 (only ONCE)
  ############################
  temperature=0.0
  port=${ports[0]}
  index=0

  echo "[T=0.0] Single run on port ${port}"

  python aguvis_uitars_plan.py \
    --model "${model_path}" \
    --agent_family "${agent_family}" \
    --input_file "${input_file}" \
    --output_file "data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl" \
    --screenshot_dir "${screenshot_dir}" \
    --level "${level}" \
    --temperature "${temperature}" \
    --port "${port}" \
    --add_template "${add_template}" \
    --reasoning "${reasoning}"  #--sem_limit 1 

  ############################
  # temperature = 1.0 (16 repetitions)
  ############################
  temperature=1.0
  total_runs=4
  runs_done=0
  round=0

  while [ ${runs_done} -lt ${total_runs} ]; do
    echo
    echo "[T=1.0] Round ${round} (runs ${runs_done} ~ $((runs_done+7)))"

    index_offset=${runs_done}
    i=0

    for port in "${ports[@]}"; do
      if [ ${runs_done} -ge ${total_runs} ]; then
        break
      fi

      index=$((index_offset + i))
      echo "  -> Run index=${index} on port=${port}"

      python aguvis_uitars_plan.py \
        --model "${model_path}" \
        --agent_family "${agent_family}" \
        --input_file "${input_file}" \
        --output_file "data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl" \
        --screenshot_dir "${screenshot_dir}" \
        --level "${level}" \
        --temperature "${temperature}" \
        --port "${port}" \
        --add_template "${add_template}" \
        --reasoning "${reasoning}" &

      i=$((i + 1))
      runs_done=$((runs_done + 1))
    done

    wait
    round=$((round + 1))
  done

  echo "Finished LEVEL=${level}"
  echo
done
