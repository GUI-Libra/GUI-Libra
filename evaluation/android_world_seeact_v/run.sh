# Model to evaluate (HuggingFace model name or path)
model_name=GUI-Libra/GUI-Libra-8B

# Output directory
output_path='./eval_results/'${model_name}
save_img_dir=${output_path}/images

# vLLM API endpoint
base_url=http://localhost:8001/v1

# Emulator gateway host(s) — add more for multi-host setups
HOST1=http://localhost:23333
HOSTS=($HOST1)

# Build emulator URLs (/emu1 through /emu15 per host)
URLS=()
for host in "${HOSTS[@]}"; do
  for i in {1..15}; do
    URLS+=("$host/emu$i")
  done
done

URLS_JOINED=$(IFS=, ; echo "${URLS[*]}")
echo "Using environment URLs: $URLS_JOINED"
echo "Model Name: ${model_name}"
echo "Output Path: ${output_path}"
echo "Hosts: ${HOSTS[@]}"

python run_suite_on_docker_mp.py \
    --env_urls ${URLS_JOINED} \
    --num_workers 15 --model ${model_name} --max_steps 20 --temperature 0.0 \
    --save_img_dir ${save_img_dir} --output_path ${output_path} --base_url ${base_url}

echo "Model Name: ${model_name}"
echo "Output Path: ${output_path}"
echo "Hosts: ${HOSTS[@]}"
