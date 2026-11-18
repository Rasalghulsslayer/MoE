#Fixation des seeds et choix du device
import torch
import random
import resource
import os
import numpy as np
import torch.distributed as dist

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("All seeds set")

def get_device(preference=None):
    if preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preference == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif preference == "cpu":
        device = torch.device("cpu")
    elif preference is None: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else: 
            device = torch.device("cpu")
    else:
        raise ValueError("Device preference not accessible, process canceled.")
    print(f"Device selected : {device}")
    return device

def mem_mo() -> float:
    """Pic de mémoire du processus en Mo (ru_maxrss est en octet sur macOS)"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

def setup_distributed():
    """Initialise le groupe de processus si lancé par torchrun, sinon rend None"""
    if "WORLD_SIZE" not in os.environ:
        return None
    
    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return (rank, world_size)

def is_main_process(dist) -> bool:
    return dist is None or dist[0] == 0