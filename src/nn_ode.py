import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(ODEFunc,self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, t, state):
		return self.net(state)

def solve_ode(func, initial_state, time_points):
	return odeint(func, initial_state, time_points)
