# Highway Environment Reinforcement Learning Project

This repository contains implementations of various reinforcement learning algorithms applied to the Highway Environment using the `highway-env` package. The project explores different approaches to autonomous driving in a simulated highway environment.

## Project Structure

```
Highway Env/
├── DQN_experiments/    # Deep Q-Network implementation and results
├── PPO_experiments/    # Proximal Policy Optimization implementation and results
├── A2C_experiments/    # Advantage Actor-Critic implementation and results
└── saved_models/      # Trained model checkpoints
```

## Implemented Algorithms

1. **Deep Q-Network (DQN)**
   - Implementation of DQN with experience replay and target networks
   - Suitable for discrete action spaces in the highway environment

2. **Proximal Policy Optimization (PPO)**
   - Implementation of PPO with clipped objective
   - Handles continuous action spaces effectively

3. **Advantage Actor-Critic (A2C)**
   - Implementation of A2C with value function baseline
   - Combines policy gradient and value-based methods

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- highway-env
- numpy
- matplotlib

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

## Results

The project compares the performance of different RL algorithms in the highway environment:

- DQN shows stable learning for discrete action spaces
- PPO demonstrates better sample efficiency
- A2C provides a good balance between complexity and performance

Detailed results and comparisons can be found in the project report.

## Model Checkpoints

Trained models are saved in the `saved_models` directory. You can load these models to evaluate their performance or continue training.

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Highway-env package developers
- OpenAI Gymnasium team
- PyTorch community
