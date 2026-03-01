dir='path/to/your/dir'
screenshot_dir='path/to/AndroidControl_images'

model_paths=(
path/to/GUI-R1-7B
)

exp_names=(
GUI-R1-7B-baseline
)

ports=(8004 8005 8006 8007)

script=guir1_plan.py
agent_family=gui_r1

for level in 'low' 'high'
do
    temperature=0.0
    for index in 0
    do
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[]}"
            exp_name="${exp_names[]}"
            port="${ports[]}"

            echo "[GUI-R1] level=${level}, index=${index}, temp=${temperature}, port=${port}"
            python ${script} \
                --agent_family ${agent_family} \
                --model ${model_path} \
                 --input_file data/500_steps_filtered_bbox_filtered_bbox_matchinstruction.json \
                --output_file data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl \
                --screenshot_dir ${screenshot_dir} \
                --level  ${level} \
                --temperature ${temperature} \
                --port ${port} \
                --max_completion_tokens 1024 \
                --sem_limit 64  &
        done
        wait
    done

    temperature=1.0
    for index in 0 1 2 3
    do
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[]}"
            exp_name="${exp_names[]}"
            port="${ports[]}"

            echo "[GUI-R1] level=${level}, index=${index}, temp=${temperature}, port=${port}"
            python ${script} \
                --agent_family ${agent_family} \
                --model ${model_path} \
                --input_file data/500_steps_filtered_bbox_filtered_bbox_matchinstruction.json \
                --output_file data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl \
                --screenshot_dir ${screenshot_dir} \
                --level  ${level} \
                --temperature ${temperature} \
                --port ${port} \
                --max_completion_tokens 1024 \
                --sem_limit 64 &
        done
        wait
    done
done
