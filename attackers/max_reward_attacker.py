import numpy as np
import torch
import torch.optim as optim


def max_reward_attack(evaluator, obs, epsilon, *args, **kwargs):
    perturbation = torch.nn.Parameter(torch.zeros_like(obs, dtype=torch.float32))
    lr_adv = 0.1
    optimizer = optim.Adam([perturbation], lr=lr_adv)
    tol = 1e-4
    eps_tol = 1e-4

    obs.requires_grad_(True)
    _action = evaluator._actor.predict(obs)
    obs_range = epsilon * torch.ones_like(obs)

    perturbation_prev = perturbation.data.detach().clone()
    loss_prev = float('inf')
    reward_weight = 1
    for i in range(20):
        obs_adv = perturbation + obs
        obs_adv.requires_grad_(True)
        adv_action = evaluator._actor.predict(obs_adv)
        if evaluator._cfgs['algo'] == 'PPOLag':
            Q_adv = evaluator._actor_critic.reward_critic(obs_adv)[0]
        elif evaluator._cfgs['algo'] == 'TD3Lag':
            Q_adv = evaluator._actor_critic.reward_critic(obs_adv, adv_action)[0]
        optimizer.zero_grad()
        loss = torch.mean(-reward_weight * Q_adv)
        loss_np = loss.item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            perturbation.clamp_(-obs_range, obs_range)

        perturbation_diff = perturbation.data.detach() - perturbation_prev
        perturbation_diff_max = torch.abs(perturbation_diff).max().item()
        perturbation_prev = perturbation.data.detach()

        if np.abs(loss_np - loss_prev) < tol and perturbation_diff_max < eps_tol:
            break
        loss_prev = loss_np

    return perturbation