# Adversarial attackers of reinforcement learning 


## Introduction
This repo is the implementation of adversarial attackers published in the paper for reinforcement learning/safe reinforcement learning.
This simulation environment is based on [safety-gymnasium](safety-gymnasium.readthedocs.io/en/latest/), and the implementation of RL algorithms is based on [Omnisafe](https://www.omnisafe.ai/en/latest/).

The implented attacers are:

1. **FGSM Attacker [1]**
2. **Random Attacker**
3. **MAD Attacker [3]**
4. **Gradient Attacker [2]**
5. **Max-Cost attacker [4]**
6. **Max-Reward attacker [4]**



## References
[1] Huang, Sandy, et al. "Adversarial attacks on neural network policies." arXiv preprint arXiv:1702.02284 (2017).

[2] Pattanaik, Anay, et al. "Robust deep reinforcement learning with adversarial attacks." arXiv preprint arXiv:1712.03632 (2017).

[3] Zhang, Huan, et al. "Robust deep reinforcement learning against adversarial perturbations on state observations." Advances in Neural Information Processing Systems 33 (2020): 21024-21037.

[4] Liu, Zuxin, et al. "On the robustness of safe reinforcement learning under observational perturbations." arXiv preprint arXiv:2205.14691 (2022).


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
