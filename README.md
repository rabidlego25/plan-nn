# Neural ODE Framework for Classical Trajectories

This project implements a Neural ODE framework to simulate and learn dynamical systems, with a specific application to planetary motion. It uses PyTorch and the `torchdiffeq` library to solve ODEs and train models for trajectory prediction.

## Features
- Simulate planetary motion using numerical ODE solvers.
- Train Neural ODE models for trajectory prediction.
- Save and visualize results using Matplotlib.
- Modular code structure for scalability and reusability.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rabidlego25/plan-nn.git
   cd plan-nn
2. Set up a Python virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
3. Install the required dependencies
   ```bash
   pip3 install -r requirements.txt

## Usage

### Simulate Planetary Motion

Run the `simulate.py` scipt to simulate planetary motion and save the results
   ```bash
   python3 src/simulate.py
* Simulation data is saved in the `data/` directory as `.npy` files.
