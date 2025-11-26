import torch
print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("built with CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
