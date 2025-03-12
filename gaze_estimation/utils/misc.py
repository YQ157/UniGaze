import datetime
import torch
import builtins
import torch.distributed as dist
from torch import inf
def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)
		
def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	builtin_print = builtins.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		force = force or (get_world_size() > 8)
		if is_master or force:
			now = datetime.datetime.now().time()
			builtin_print('[{}] '.format(now), end='')  # print with time stamp
			builtin_print(*args, **kwargs)

	builtins.print = print
	

def all_reduce_mean(x):
	world_size = get_world_size()
	if world_size > 1:
		x_reduce = torch.tensor(x).cuda()
		# x_reduce = x.clone()
		dist.all_reduce(x_reduce)
		x_reduce /= world_size
		return x_reduce.item()
	else:
		return x
	


def gather_tensors(tensor):
	# Gather tensors across all processes
	gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
	dist.all_gather(gather_list, tensor)
	# return torch.cat(gather_list, dim=0)

	# Stack tensors along a new dimension (each GPU result along dimension 0)
	tensor_gather = torch.stack(gather_list, dim=0)
	# Transpose to get them in the correct order (swap dimensions 0 and 1)
	tensor_gather = tensor_gather.transpose(0, 1)
	# Reshape back into a 2D tensor with the correct order
	tensor_gather = tensor_gather.reshape(-1, tensor.size(-1))
	return tensor_gather




def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
	return total_norm


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
		self._scaler.scale(loss).backward(create_graph=create_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = get_grad_norm_(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)