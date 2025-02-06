import torch
import numpy as np

def random_attack(evaluator, obs, epsilon, *args, **kwargs):
    # the original obs
    _obs = obs
    # the range of the perturbation, initialized as epsilon
    obs_range = torch.full_like(obs, epsilon)

    # Generate random attack using normal distribution
    attack = torch.normal(mean=0, std=obs_range)

    # Calculate the lower and upper bounds of the perturbed state
    lower_bound = obs - obs_range
    upper_bound = obs + obs_range

    # Apply the attack and clip the state
    perturbed_obs = torch.clip(obs + attack, min=lower_bound, max=upper_bound)

    # Return the perturbation applied
    return perturbed_obs - obs