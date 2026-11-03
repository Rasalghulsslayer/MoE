#Fixation des seeds et choix du device
import torch
import random
import resource
import numpy as np

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