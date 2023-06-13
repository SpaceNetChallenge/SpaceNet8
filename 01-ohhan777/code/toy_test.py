import os
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train():
    if RANK in [-1, 0]:
        print(f"Hi! I am process {RANK}. World size is {WORLD_SIZE}")
    print(f"Process {RANK} goes to bed.")
    sleep_time = random.randrange(1,5)
    time.sleep(sleep_time)
    print(f"Process {RANK} slept {sleep_time} seconds")
    


def main():
    random.seed(time.time())
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    train()
    if WORLD_SIZE > 1 and RANK == 0:
        dist.destroy_process_group()

    
if __name__ == "__main__":
    main()