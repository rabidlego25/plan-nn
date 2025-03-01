# Neural ODE Framework for Classical Trajectories

This project implements a Neural ODE framework to simulate and learn dynamical systems, with a specific application to planetary motion. It uses PyTorch and the `torchdiffeq` library to solve ODEs and train models for trajectory prediction.

## Features
- Simulate planetary motion using numerical ODE solvers.
- Train Neural ODE models for trajectory prediction.
- Save and visualize results using Matplotlib.
- Modular code structure for scalability and reusability.

#### Please note this code requires python be versioned less than 3.11 to comply with the torch ecosystem

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rabidlego25/plan-nn.git
   cd plan-nn
   ```
2. Set up a Python virtual environment:
   ```bash
   python -m venv ../myenv
   source ../myenv/bin/activate
   ```
3. Install the required dependencies
   ```bash
   poetry install
   ```

## Usage

### Simulate Planetary Motion

Run the `simulate.py` scipt to simulate planetary motion and save the results
   ```bash
   python src/simulate.py
   ```
- *Simulation data is saved in the `data/` directory as `.npy` files.*

### Train the Neural Network

- *Use the `train_ode.py` script to train a Neural ODE model*
   ```bash
   python src/train_ode.py
   ```
- *The trained model is saved as `trained_ode_func.pth`*

##  Model Explanation

### Description of key files in [`src`](src) directory

- [`src/train_ode.py`](src/train_ode.py) &rarr; Trains a neural ODE model using PyTorch and `torchdiffeq` to predict software version compatibility or dependencies.
- [`src/nn_ode.py`](src/nn_ode.py) &rarr; Defines a neural ODE function using a multi-layer perceptron (MLP) for time-dependent predictions.
- [`src/simulate.py`](src/simulate.py) &rarr; Implements a physics-based simulation of orbital motion and preprocesses data for analysis.
- [`src/specs.py`](src/specs.py) &rarr; Gathers system information, including CPU, GPU, and PyTorch details, to optimize software installation.

### Benefit of using .npy files

- `.npy` files are faster compared to text-based formats like CSV.
- Represents the data in native binary form - avoiding the overhead of formatting and parsing
- Saving numerical data as text (ie CSV), floating-point values get rounded due to formatting
- The data type `int32`, `float64`, etc are stored alongside the data preserving data integrity
