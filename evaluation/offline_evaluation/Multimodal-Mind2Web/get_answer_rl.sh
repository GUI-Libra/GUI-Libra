dir=path/to/your/dir
relative_coord=0 # 0: absolute coordinates for Qwen2.5 vl based models, 1: relative coordinates for Qwen3 vl based models

echo "Evaluating pass@1 in directory: ${dir}"
for key in 'cross_task'  'cross_website'   'cross_domain'
do
    echo "Evaluating pass@1 for key: ${key}"
    plan_file=${dir}/${key}/plan_0_temperature0.0.jsonl
    python eval_rl.py --plan_file ${plan_file} --relative_coord ${relative_coord}
done

for n in 4  
do
    echo "Evaluating pass@${n} plans in directory: ${dir}"
    for key in 'cross_task'  'cross_website'  'cross_domain'
    do
        echo "Evaluating pass@${n} plans for key: ${key}"
        plan_file=${dir}/${key}/plan_0_temperature1.0.jsonl
        python eval_rl_bestofn.py --plan_file ${plan_file} --file_num ${n} --relative_coord ${relative_coord}
    done
done
