# Deep Q-Network (DQN) Experiments

This directory contains the implementation and experiments of Deep Q-Network (DQN) algorithm on the Highway Environment.

## Directory Structure

```
DQN_experiments/
├── DQN_experiments.ipynb    # Main implementation and training notebook
├── tensorboard_logs/       # Training metrics and visualization logs
└── videos/                 # Recorded agent performance videos
```

## Implementation Details

The DQN implementation includes:
- Experience replay buffer for storing and sampling transitions
- Target network for stable Q-value estimation
- ε-greedy exploration strategy
- Double DQN architecture to reduce overestimation bias

## Network Architecture

- Input: State representation from highway environment
- Hidden layers: Fully connected layers with ReLU activation
- Output: Q-values for each possible action

## Training Parameters

- Learning rate: 0.001
- Discount factor (γ): 0.99
- Replay buffer size: 100,000
- Batch size: 32
- Target network update frequency: Every 1000 steps
- ε-greedy: Initial ε = 1.0, final ε = 0.05

## Usage

1. Open `DQN_experiments.ipynb` in Jupyter Notebook or JupyterLab
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

- DQN effectively learns lane-changing and speed adjustment behaviors
- Stable learning curve after approximately 100k steps
- Successfully avoids collisions while maintaining desired speed 