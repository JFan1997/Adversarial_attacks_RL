# Adversarial attackers of reinforcement learning 


## Introduction
This repo is the implementation of adversarial attackers published in the paper for reinforcement learning/safe reinforcement learning.
This simulation environment is based on [safety-gymnasium](safety-gymnasium.readthedocs.io/en/latest/), and the implementation of RL algorithms is based on [Omnisafe](https://www.omnisafe.ai/en/latest/).

The implented attacers are:

1. **FGSM Attacker [1]**
2. **Random Attacker**
3. **MAD Attacker [3]**
4. **Gradient Attacker [2]**
5. **Max-Cost attacker [5]**
6. **Max-Reward attacker [6]**



## References
[1] Adversarial Attacks on Neural Network Policies.

[2] Pattanaik, A., Tang, Z., Liu, S., Bommannan, G., Chowdhary, G.: Robust deep reinforcement learning with adversarial attacks. arXiv preprint arXiv:1712.03632 (2017)

[3] Robust Deep Reinforcement Learning against  Adversarial Perturbations on State Observations

[4] On the Robustness of Safe Reinforcement Learning under Observational Perturbations.


## Installation
```bash
pip install -r requirements.txt
```

## Train the policy
To train the victim PPOLag policy of PointCircle simulation, run the following command:
```bash
python trainPointCircle.py
```

## Usage
To run the attack of PointCircle simulation, run the following command:
```bash
python run_all_attackers_circle.py
```
