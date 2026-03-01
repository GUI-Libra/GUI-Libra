model='gpt-4o'
# model='o4-mini'
for level in 'high'  'low'
do 
    for index in 0 1 2 3
    do
        python extract_grounding_query.py \
        --sample_file data/500_steps.json \
        --input_file data/${model}-plan-repeat/plan_${level}_${index}_temperature1.jsonl \
        --output_file data/${model}-plan-repeat/query_${level}_${index}_temperature1.jsonl \
        --screenshot_dir path/to/AndroidControl_images 
    done
done
