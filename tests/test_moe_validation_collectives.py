from unittest.mock import patch
import torch
from torch import nn
from ultralytics.nn.modules.moe.loss import MoELoss,differentiable_balance_loss,should_reduce_ddp
def test_eval_and_no_grad_local():
 m=nn.Linear(2,2).eval()
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2),patch('torch.distributed.all_reduce') as r:
  assert not should_reduce_ddp(m)
  with torch.no_grad(): differentiable_balance_loss(torch.tensor([[.6,.4]]),torch.tensor([.5,.5]),2,reduce_ddp=should_reduce_ddp(m))
  r.assert_not_called()
def test_train_grad_global():
 m=nn.Linear(2,2).train()
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2): assert should_reduce_ddp(m)
def test_moe_loss_eval_hidden_usage_local():
 loss=MoELoss(num_experts=2).eval();p=torch.tensor([[.7,.3]])
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2),patch('torch.distributed.all_reduce') as r:
  with torch.no_grad(): loss(p,p.log(),torch.tensor([[0]]))
  r.assert_not_called()
