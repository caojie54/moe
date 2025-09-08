export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

model=Qwen3-14B

experts=16

python train.py @configs/${model}_mocorelora_math14k_train_exp${experts}.config

python test_math.py @configs/${model}_mocorelora_math14k_test_exp${experts}.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${model}-mocorelora-exp${experts}-math-14k/predictions/addsub_responses.jsonl
