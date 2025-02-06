import numpy as np
import torch
from scipy.optimize import minimize, Bounds


def MAD_attacker(evaluator, obs, epsilon,*args,**kwargs):
    # evaluator, MAD, epsilon, total_epoch, seed
    original_action = evaluator._actor.predict(obs, deterministic=True)

    def objective(delta_state):
        # Compute the perturbed state
        perturbed_state = obs + torch.tensor(delta_state, dtype=torch.float32)
        # Predict action for the perturbed state
        perturbed_action = evaluator._actor.predict(perturbed_state, deterministic=True)
        # Calculate action difference
        action_diff = original_action - perturbed_action
        # Maximize action difference by minimizing negative squared difference
        return -np.sum(np.power(action_diff.detach().numpy(), 2))

    x_start = np.zeros_like(obs)
    bounds = Bounds(-epsilon * np.ones_like(obs), epsilon * np.ones_like(obs))

    result = minimize(objective, x_start, method='trust-constr', bounds=bounds, tol=1e-4)
    op_action = result.x
    # print('returned op action',op_action)
    return op_action
