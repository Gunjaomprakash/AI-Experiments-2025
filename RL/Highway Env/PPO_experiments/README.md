# Proximal Policy Optimization (PPO) Experiments

This directory contains the implementation and experiments of Proximal Policy Optimization (PPO) algorithm on the Highway Environment.

## Directory Structure

```
PPO_experiments/
├── PPO_experiments.ipynb    # Main implementation and training notebook
├── tensorboard_logs/       # Training metrics and visualization logs
└── videos/                 # Recorded agent performance videos
```

## Implementation Details

The PPO implementation includes:
- Actor-Critic architecture with shared network layers
- Clipped surrogate objective function
- Generalized Advantage Estimation (GAE)
- Value function loss clipping
- Entropy bonus for exploration

## Network Architecture

- Input: State representation from highway environment
- Shared layers: Fully connected layers with ReLU activation
- Policy head: Outputs action distribution parameters
- Value head: Outputs state value estimation

## Training Parameters

- Learning rate: 3e-4
- Clip range: 0.2
- Value function coefficient: 0.5
- Entropy coefficient: 0.01
- GAE lambda: 0.95
- Number of epochs per update: 10
- Batch size: 64

## Usage

1. Open `PPO_experiments.ipynb` in Jupyter Notebook or JupyterLab
2. Follow the notebook cells for:
   - Environment setup
   - Model training
   - Evaluation
   - Visualization

## Results Visualization

- Training metrics can be viewed using TensorBoard:
```bash
tensorboard --logdir=tensorboard_logs
```
- Performance videos are saved in the `videos` directory

## Key Findings

- PPO shows better sample efficiency compared to DQN
- Smooth and natural driving behavior
- Good balance between exploration and exploitation
- Robust performance across different traffic scenarios 