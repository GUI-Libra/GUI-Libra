dir='2026_2_3'
add_template=1
reasoning=1
screenshot_dir='path/to/AndroidControl_images'

model_paths=(
path/to/your/model
)
 
 

exp_names=(
Qwen3VL_8B_fromcpt636_weightedSFT_ratio2_mixnoreasoning_grpo_ngsx2.0mean_0.5-1_wkl0.001_step600
)


ports=(8000 8001 8002 8003 8004 8005 8006 8007)

# ports=(8001)

for level in   'high' 'low'   
do 
    
    temperature=0.0
    for index in 0
    do 
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[$i]}"
            exp_name="${exp_names[$i]}"
            port="${ports[$i]}"
            echo "Processing level: ${level}, index: ${index}, model_path: ${model_path}"
            python guilibra_plan.py --model ${model_path} \
                    --input_file data/500_steps_filtered.json \
                    --output_file data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl \
                    --screenshot_dir ${screenshot_dir} \
                    --level  ${level}  --temperature ${temperature} \
                    --port ${port}  --add_template ${add_template} --reasoning     ${reasoning}  &
        done
        wait
    done


    temperature=1.0
    for index in 0 1 2 3 
    do 
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[$i]}"
            exp_name="${exp_names[$i]}"
            port="${ports[$i]}"
            echo "Processing level: ${level}, index: ${index}, model_path: ${model_path}"
            python guilibra_plan.py --model ${model_path} \
                    --input_file data/500_steps_filtered.json \
                    --output_file data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl \
                    --screenshot_dir ${screenshot_dir} \
                    --level  ${level}  --temperature ${temperature} \
                    --port ${port}  --add_template ${add_template} --reasoning     ${reasoning}  &
        done
        wait
    done
done

