"""torchrun smoke for CPU/gloo or CUDA/NCCL sparse-branch DDP."""
import argparse, os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
class Sparse(nn.Module):
 def __init__(self):
  super().__init__(); self.shared=nn.Linear(4,4); self.experts=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
  self.register_buffer("cpu_diagnostic", torch.tensor(0.), persistent=False)
 def forward(self,x,rank): return self.experts[rank%2](self.shared(x)).sum()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--backend',choices=['gloo','nccl'],default='gloo'); a=ap.parse_args()
 rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
 if a.backend=='nccl':
  if not torch.cuda.is_available() or local>=torch.cuda.device_count(): raise RuntimeError('invalid CUDA local rank')
  torch.cuda.set_device(local); device=torch.device('cuda',local)
 else: device=torch.device('cpu')
 dist.init_process_group(a.backend,timeout=timedelta(seconds=60))
 try:
  model=Sparse().to(device); model.cpu_diagnostic=torch.tensor(float(rank))
  ddp=DDP(model,device_ids=[local] if device.type=='cuda' else None,output_device=local if device.type=='cuda' else None,find_unused_parameters=True,broadcast_buffers=False,static_graph=False)
  opt=torch.optim.SGD(ddp.parameters(),lr=.1); opt.zero_grad(); ddp(torch.ones(2,4,device=device),rank).backward(); opt.step()
  flat=torch.cat([p.detach().reshape(-1) for p in ddp.module.parameters()]); gathered=[torch.empty_like(flat) for _ in range(world)]; dist.all_gather(gathered,flat)
  assert all(torch.allclose(gathered[0],v) for v in gathered[1:]); assert ddp.module.cpu_diagnostic.device.type=='cpu'
  if rank==0: print(f'DDP sparse smoke passed: backend={a.backend}, world_size={world}')
 finally: dist.destroy_process_group()
if __name__=='__main__': main()
