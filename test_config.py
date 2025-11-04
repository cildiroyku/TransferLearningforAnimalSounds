import os
import yaml
import torch

#Load YAML configuration
config_path = "configs/base.yaml"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

print("Config loaded successfully!")
print(f"Project name: {cfg['project']['name']}")
print(f"Sample rate: {cfg['data']['sample_rate']}")
print(f"Device preference: {cfg['project']['device']}")
print()

#Check folders mentioned in config
required_dirs = [
    cfg["data"]["gtzan_dir"],
    cfg["data"]["esc50_dir"],
    cfg["data"]["save_embeddings_dir"],
]

for d in required_dirs:
    os.makedirs(d, exist_ok=True)  # creates empty ones if missing
    print(f"Verified folder: {d}")

print()

#GPU check
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not detected — using CPU")
