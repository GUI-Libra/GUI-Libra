sample_file=data/500_steps_filtered.json

dir=path/to/your/generated_plans
relative_coord=0

for mode in high  low
do
    echo "Evaluating plans in directory: ${dir}"
    echo "Best of 1 performance for mode: ${mode}"
    plan_file=${dir}/plan_${mode}_0_temperature0.0.jsonl
    python eval_rl.py --sample_file ${sample_file} --plan_file ${plan_file} --relative_coord ${relative_coord}

    for num in 4 
    do
        echo "Best of ${num} performance for mode: ${mode}"
        plan_file=${dir}/plan_${mode}_0_temperature1.0.jsonl
        python eval_best_of_n.py --sample_file ${sample_file} --plan_file ${plan_file} --file_num ${num} --relative_coord ${relative_coord}
    done
done