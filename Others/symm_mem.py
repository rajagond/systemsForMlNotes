"""
https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798
"""
import os
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(f"cuda:{rank}")
dist.init_process_group("nccl")

prev_rank = (rank - 1) % world_size
next_rank = (rank + 1) % world_size

# Allocate a tensor
t = symm_mem.empty(4096, dtype=torch.bfloat16, device=f"cuda:{rank}")

# Establish symmetric memory and obtain the handle
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
peer_buf = hdl.get_buffer(next_rank, t.shape, t.dtype)
print(dir(hdl))
# Pull
t.fill_(rank)
hdl.barrier(channel=0)
pulled = torch.empty_like(t)
pulled.copy_(peer_buf)
hdl.barrier(channel=0)
assert pulled.eq(next_rank).all()

# Push
hdl.barrier(channel=0)
to_push = torch.full_like(t, rank)
peer_buf.copy_(to_push)
hdl.barrier(channel=0)
assert t.eq(prev_rank).all()

# Direct
t.fill_(rank)
hdl.barrier(channel=0)
torch.add(peer_buf, rank, out=peer_buf)
hdl.barrier(channel=0)
assert t.eq(rank + prev_rank).all()

# Raw pointers for CUDA/Triton kernels
hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs

if rank == 0:
    print(f"Rank {rank}, type: {type(hdl.buffer_ptrs)}, "
          f"multicast_ptr: {type(hdl.multicast_ptr)}, "
            f"signal_pad_ptrs: {type(hdl.signal_pad_ptrs)}")

    print(f"Rank {rank} completed successfully.")
    print(f"Rank {rank} buffer_ptrs: {hdl.buffer_ptrs}, multicast_ptr: {hdl.multicast_ptr}, signal_pad_ptrs: {hdl.signal_pad_ptrs}")
    print(f"Rank {rank} get_rank: {hdl.rank}, get_world_size: {hdl.world_size}", flush=True)

hdl.barrier(channel=0)

t.fill_(rank)
hdl.barrier(channel=0)
print(f"Rank {rank} tensor after barrier: {t}", flush=True)
if rank == 0:
    print(f"Rank {rank} hdl after barrier: {hdl}", flush=True)
hdl.barrier(channel=0)
print(f"Rank {rank} tensor after multicast: {t}", flush=True)
