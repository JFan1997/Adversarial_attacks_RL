import numpy as np
import torch


def sample_beta_noise(size, alpha, beta):
    """
    Sample noise from Beta(α, β) distribution and shift to range [-0.5, 0.5].

    Args:
        size: Tuple, the shape of the noise tensor (e.g., (batch_size, num_features)).
        alpha: Shape parameter α of the Beta distribution.
        beta: Shape parameter β of the Beta distribution.

    Returns:
        Tensor of Beta-distributed noise in the range [-0.5, 0.5].
    """
    # Sample from Beta(α, β)
    beta_dist = torch.distributions.Beta(alpha, beta)
    noise = beta_dist.sample(size)

    # Shift to range [-0.5, 0.5]
    noise_shifted = noise - 0.5

    return noise_shifted


def gradient_attack(evaluator, obs, epsilon, *args, **kwargs):
    # action = evaluator._actor.predict(obs, deterministic=True)
    # the original action
    # print('this is obs type first',obs.dtype)
    obs.requires_grad_(True)
    algorithm = evaluator._cfgs['algo']
    _action = evaluator._actor.predict(obs)
    # the original obs
    # _obs = obs
    perturbed_obs = obs.clone()
    # the updated action, initialized as the original action
    action = evaluator._actor.predict(obs, deterministic=True)
    # obs_range = _obs / np.linalg.norm(_obs) * epsilon * math.sqrt(44)
    # the range of the perturbation, initialized as epsilon
    obs_range = torch.tensor([epsilon])
    # original Q value
    if algorithm == 'PPOLag':
        Q = evaluator._actor_critic.reward_critic(obs)[
            0]
    elif algorithm == 'TD3Lag':
        Q = evaluator._actor_critic.reward_critic(obs, action)[
            0]
    Q.backward()
    obs_grad = obs.grad
    obs_grad_dir = obs_grad.sign()
    attack = None
    Q_optimal = Q
    for i in range(20):
        noise = sample_beta_noise(obs.size(), 1, 1)
        perturbed_obs.requires_grad_(True)
        perturbed_obs = perturbed_obs - noise * obs_grad_dir
        perturbed_obs = torch.clamp(perturbed_obs, obs - obs_range, obs + obs_range)
        adv_action = evaluator._actor.predict(perturbed_obs)
        if algorithm == 'PPOLag':
            Q_adv = evaluator._actor_critic.reward_critic(obs)[0]
        elif algorithm == 'TD3Lag':
            Q_adv = evaluator._actor_critic.reward_critic(obs, adv_action)[0]
    attack = perturbed_obs.detach() - obs
    if attack is not None:
        # print('this is the attack', attack)
        return attack
    else:
        return np.zeros_like(obs.detach().numpy())
