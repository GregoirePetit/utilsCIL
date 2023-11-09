# check if GPU(s) is/are available
import torch
def is_GPU_available():
    return torch.cuda.is_available()

if __name__ == '__main__':
    if is_GPU_available():
        print("GPU(s) is/are available.")
        print("Number of GPU(s):", torch.cuda.device_count())
        print("Name of GPU(s):", torch.cuda.get_device_name(0))
    else:
        print("GPU(s) is/are not available.")
        