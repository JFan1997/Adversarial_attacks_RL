import torch


def loss_pi(evaluator, obs):
    action = evaluator._actor_critic.actor.predict(obs, deterministic=True)
    # 最小化-reward critic的Q值，相当于最大化reward
    loss_r = -evaluator._actor_critic.reward_critic(obs, action)[0]
    # 最小化cost critic的Q值，相当于最小化cost
    loss_q_c = evaluator._actor_critic.cost_critic(obs, action)[0]
    # 乘以拉格朗日乘子，得到最终的cost loss
    # loss_c = evaluator._lagrange.lagrangian_multiplier.item() * loss_q_c
    # print('loss_r:', loss_r, 'loss_q_c:', loss_q_c, 'loss_c:', loss_c)
    # 返回最终的loss
    return (loss_r + loss_q_c).mean()


def FGSM_attacker(evaluator, obs, epsilon, *args, **kwargs):
    # Clone obs to avoid modifying original input tensor
    obs_adv = obs.clone().detach().requires_grad_(True)
    loss = loss_pi(evaluator, obs_adv)
    # print('this is loss', loss)
    # loss = evaluator._loss_pi(obs_adv)
    loss.backward()
    perturbation = epsilon * obs_adv.grad.sign()
    # print('this is the perturbation', perturbation)
    obs_adv_final = obs + perturbation
    obs_adv_final = torch.clamp(obs_adv_final, obs - epsilon, obs + epsilon)
    attack = obs_adv_final - obs
    return attack.detach()
