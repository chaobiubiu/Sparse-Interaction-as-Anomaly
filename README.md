# Multi-Agent Sparse Interaction Modeling is an Anomaly Detection Problem

Code for the paper "Multi-Agent Sparse Interaction Modeling is an Anomaly Detection Problem" submitted to ICASSP 2024.

This repository develops SIA algorithm on StarCraft Multi-Agent Challenge benchmark. In addition, we compare it with 
multiple baselines including MAVEN, QTRAN and QPLEX. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the approach in the paper, run this command:

```train
python main.py
```

You can select the training algorithm by setting ```--config='sia' or other available choices```


## Hyper-parameters

To modify the hyper-parameters of algorithms and environments, refer to:

```
src/config/algs/sia.yaml
src/config/default.yaml
```
```
src/config/envs/sc2.yaml
```

## Note

This repository is developed based on PyMARL. And we have cited the SMAC paper in our work.