import torch

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA not available. Training will run on CPU.")
