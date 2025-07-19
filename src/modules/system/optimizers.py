import torch

class CUDAAccelerator:
    def __init__(self, use_tensor_cores=True):
        torch.backends.cuda.matmul.allow_tf32 = use_tensor_cores
        torch.backends.cudnn.allow_tf32 = use_tensor_cores
        self.memory_map = 'unified'
        logger.info("⚡ NVIDIA Tensor Cores aktiviert (FP16/FP32-Mixed Precision)")