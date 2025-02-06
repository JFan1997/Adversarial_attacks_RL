import os

import omnisafe
import torch


def attack_runner(evaluator, attack_strategy, epsilon, total_epoch, *args, **kwargs):
    reach = 0
    violate = 0
    agent_positions = []
    for epoch in range(total_epoch):
        print('current epoch is:', epoch)
        epoch_positions = []
        obs, info = evaluator._env.reset()
        total_step = 0
        done = False
        trun = False
        while not done and not trun:
            attack = attack_strategy(evaluator, obs, epsilon, *args, **kwargs)
            attack = torch.tensor(attack).float()
            # change the observation space
            obs = attack + obs
            action = evaluator._actor.predict(obs, deterministic=True)
            obs, reward, cost, done, trun, info = evaluator._env.step(action)
            agent_pos = evaluator._env.unwrapped_env.task.agent.pos
            # pos = [agent_pos[0], agent_pos[1]]
            # _agent_x, _agent_y = pos[0], pos[1]
            epoch_positions.append(agent_pos)
            total_step += 1
            if cost != 0:
                print('cost !=0, voiolate+=1', cost, 'epoch:', epoch)
                violate += 1
                break
            if trun:
                print('trun reached max steps: {}'.format(total_step))
                reach += 1
        agent_positions.append(epoch_positions)
    print(
        f'{attack_strategy.__name__} attack violation: {violate}, reach: {reach}, '
        f'violation prob: {violate / total_epoch}, reach prob: {reach / total_epoch}'
    )
    return agent_positions, violate, reach


def get_evaluator(log_dir, render_mode='human'):
    scan_dir = os.scandir(os.path.join(log_dir, 'torch_save'))
    it = []
    for item in scan_dir:
        it.append(item)
    it.sort(key=lambda item: int(item.name.split('-')[1].split('.')[0]))
    item = it[-1]
    evaluator = omnisafe.Evaluator(render_mode=render_mode)
    print('initialized evaluator env type', type(evaluator._env))
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        print('find model!')
        evaluator.load_saved(
            save_dir=log_dir,
            model_name=item.name,
            camera_name='track',
            width=256,
            height=256,
            render_mode=render_mode
        )
    print('initialized loaded evaluator env type', type(evaluator._env))
    return evaluator
