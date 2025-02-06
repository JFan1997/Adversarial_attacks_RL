import numpy as np
import torch
import torch.optim as optim

# evaluator, attacker, epsilon, mlp_model, total_epoch,
def max_cost_attack(evaluator, obs, epsilon, *args, **kwargs):
    perturbation = torch.nn.Parameter(torch.zeros_like(obs, dtype=torch.float32))
    lr_adv = 0.1
    optimizer = optim.Adam([perturbation], lr=lr_adv)
    tol = 1e-4
    eps_tol = 1e-4
    cost_weight = 1.0
    obs.requires_grad_(True)
    _action = evaluator._actor.predict(obs)
    obs_range = epsilon * torch.ones_like(obs)

    perturbation_prev = perturbation.clone().detach()
    loss_prev = float('inf')

    for i in range(20):
        obs_adv = (perturbation + obs).detach().requires_grad_(True)
        adv_action = evaluator._actor.predict(obs_adv)
        if evaluator._cfgs['algo'] == 'PPOLag':
            Q_adv = evaluator._actor_critic.cost_critic(obs_adv)[0]
        elif evaluator._cfgs['algo'] == 'TD3Lag':
            Q_adv = evaluator._actor_critic.cost_critic(obs_adv, adv_action)[0]
        optimizer.zero_grad()
        loss = torch.mean(-cost_weight * Q_adv)
        loss_np = loss.item()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation.clamp_(-obs_range, obs_range)

        perturbation_diff = perturbation - perturbation_prev
        perturbation_diff_max = torch.abs(perturbation_diff).max().item()
        perturbation_prev = perturbation.clone().detach()

        if np.abs(loss_np - loss_prev) < tol and perturbation_diff_max < eps_tol:
            break
        loss_prev = loss_np

    return perturbation
