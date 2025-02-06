import logging
import os
import sys

from attackers.gradient_attacker import gradient_attack
from attackers.mad_attacker import MAD_attacker
from attackers.max_cost_attacker import max_cost_attack
from attackers.max_reward_attacker import max_reward_attack
from attackers.random_attacker import random_attack
from utils import attack_runner


env_id = 'SafetyPointCircle1-v0'

alg = 'TD3Lag'

from utils import get_evaluator

# 配置 logging
logging.basicConfig(filename='attack_results_circle.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

log_dir = './runs/PPOLag-{SafetyPointCircle1-v0}/seed-000-2025-02-06-13-35-09'

evaluator = get_evaluator(log_dir=log_dir)
total_epoch = 20

epsilons = [0.01, 0.02, 0.05, 0.1]
attackers = [max_reward_attack, max_cost_attack, MAD_attacker, gradient_attack, random_attack]

for attacker in attackers:
    print('begin {} attack.....'.format(attacker.__name__))
    for epsilon in epsilons:
        print('attack power:', epsilon)
        # evaluator, attack_strategy, epsilon, total_epoch, seed, *args, ** kwargs
        # evaluator, attack_strategy, epsilon, total_epoch, seed, *args, ** kwargs
        violate, reach = attack_runner(evaluator, attacker, epsilon, total_epoch)
        logging.info(
            f'epsilon: {epsilon}, {attacker.__name__} attack violation: {violate}, total: {total_epoch}, reach: {reach}, '
            f'violation prob: {violate / total_epoch}, reach prob: {reach / total_epoch}')
    logging.info('----------------------------------------------')
