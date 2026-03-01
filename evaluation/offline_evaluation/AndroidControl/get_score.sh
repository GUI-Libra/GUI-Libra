
# sample_file=data/500_steps.json
sample_file=data/500_steps_filtered.json
mode='high'

dir='data/path/to/plan_results'
for mode in 'high' 'low'
do
  plan_file=${dir}/plan_${mode}_0_temperature0.0.jsonl
  ans_file=${dir}/ans_${mode}_0_uground_temperature0.0.jsonl
  python eval.py --sample_file ${sample_file} --plan_file ${plan_file} \
    --ans_file ${ans_file}
done


