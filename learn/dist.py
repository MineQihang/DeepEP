import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 定义子组（前一半进程）
    subgroup_ranks = list(range(world_size // 2))
    
    # 所有进程都必须调用 new_group（哪怕自己不在里面）
    pg = dist.new_group(ranks=subgroup_ranks)

    tensor = torch.tensor([rank], device='cuda')
    
    if rank in subgroup_ranks:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
        print(f"[Rank {rank}] reduced value = {tensor.item()}")
    else:
        print(f"[Rank {rank}] not in group, tensor = {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
