'''
GPU configuration
'''

import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use
# # os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3";   
# os.environ["CUDA_VISIBLE_DEVICES"]="1";   
# # os.environ["CUDA_VISIBLE_DEVICES"]="1"; 
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# # bugfix for "RuntimeError: received 0 items of ancdata" with Mon.ai
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

