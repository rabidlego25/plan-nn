import torch
import torch.nn as nn
from torchdiffeq import odeint
from nn-ode import ODEFunc

# hyperparameters
input_dim = 4
hidden_dim = 32
output_dim = 4
learning_rate = 1e-3
epochs = 1000

# init ODE function
ode_func = ODEFunc(input_dim, hidden_dim, output_dim)

# define optimizer and loss
optimizer = torch.optim.Adam(ode_func.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# example ground truth data
time_points = torch.linspace(0,10,100)
ground_truth = torch.zeros((100,4))
initial_state = torch.tensor([1.0,0.0,0.0,1.0],dtype=torch.float32).unsqueeze(0)

# training loop
for epoch in range(epochs):
	optimizer.zero_grad()

	# sole ODE
	predicted_trajectory = odeint(ode_func, initial_state, time_points)

	# compute loss
	loss = loss_fn(predicted_trajectory.squeeze(), ground_truth)
	loss.backward()
	optimizer.step()

	if epoch % 100 == 0:
		print(f'Epoch {epoch}, Loss: {loss.item()}')

# save trained model
torch.save(ode_func.state_dict(), 'trained_ode_func.pth')
print("Model saved as 'trained_ode_func.pth'")
