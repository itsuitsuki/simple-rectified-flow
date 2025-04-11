import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set seed for everything
def seed_everything(seed: int):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed_everything(42)