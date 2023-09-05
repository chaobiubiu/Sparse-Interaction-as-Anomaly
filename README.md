# Multi-Agent Sparse Interaction Modeling is an Anomaly Detection Problem

Code for the paper "Multi-Agent Sparse Interaction Modeling is an Anomaly Detection Problem" submitted to ICASSP 2024.

This repository develops SIA algorithm based on tabular Q-learning and compare it with multiple exploration baselines 
including EITI, MAX Entropy and Epsilon Greedy on a didactic multi-agent task named hallway.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the approach in the paper, run this command:

```train
python I2e.py
```

Here ```I2e``` refers to our approach```SIA``` in the submitted paper.

For baselines ```EITI``` and ```Max Entropy```, run commands as follows:

```train
python train_eiti.py
python train_maxeq.py
```

## Hyper-parameters

To modify the hyper-parameters of algorithms, please refer to ```I2e.py```:

```
parser.add_argument("--lr_vae", type=float, default=1e-3, help="learning rate of vae")
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--state_embedding_size", type=int, default=10)
parser.add_argument("--action_embedding_size", type=int, default=10)
parser.add_argument("--bonus_weight", type=float, default=0.1)
```

## Note

For faster learning, we initialize each agent's Q table with pre-learned single-agent optimal
policies.

Specifically, ```li_Q_value1.npy``` for agent 1 and ```li_Q_value2.npy``` for agent 2.

Please set ```args.use_cuda=True``` for normal training of all algorithms.
