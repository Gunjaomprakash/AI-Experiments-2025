# Advantage Actor-Critic (A2C) Experiments

This directory contains the implementation and experiments of Advantage Actor-Critic (A2C) algorithm on the Highway Environment.

## Directory Structure

```
A2C_experiments/
├── A2C_experiments.ipynb    # Main implementation and training notebook
├── tensorboard_logs/       # Training metrics and visualization logs
└── videos/                 # Recorded agent performance videos
```

## Implementation Details

The A2C implementation includes:
- Separate networks for actor (policy) and critic (value)
- N-step returns for advantage estimation
- Entropy regularization for exploration
- Parallel environment sampling
- Gradient clipping for stability

## Network Architecture

### Actor Network
- Input: State representation from highway environment
- Hidden layers: Fully connected layers with ReLU activation
- Output: Action probabilities (policy)

### Critic Network
- Input: State representation from highway environment
- Hidden layers: Fully connected layers with ReLU activation
- Output: State value estimation

## Training Parameters

- Learning rate: 0.0007
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Max gradient norm: 0.5
- N-step returns: 5
- Number of environments: 8
- Discount factor (γ): 0.99

## Usage

1. Open `A2C_experiments.ipynb` in Jupyter Notebook or JupyterLab
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

- A2C provides a good balance between sample efficiency and stability
- Effective in both continuous and discrete action spaces
- Parallel environment sampling improves training speed
- Consistent performance across different initial conditions 