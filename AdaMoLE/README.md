## Installation

```bash
# Navigate to the AdaMoLE directory
cd AdaMoLE

# Install required dependencies
pip install -r requirements.txt

cd transformers
pip install -e .[torch]
```

## Usage

```bash
# Train the model
python train.py @configs/qwen3-8b_lora_math14k_train.config

# Test the model
python test.py @configs/qwen3-8b_lora_math14k_train.config
```

## Citation
