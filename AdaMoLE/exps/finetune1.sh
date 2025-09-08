export CUDA_VISIBLE_DEVICES="1"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

python train.py @configs/Qwen3-8B_mocorelora_math14k_train_exp32.config

python test_math.py @configs/Qwen3-8B_mocorelora_math14k_test_exp32.config
