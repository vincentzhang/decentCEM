import os
import sys
import json
from subprocess import call
from utils.sweeper import Sweeper


def cfg_string(cfg):
    ks = sorted(cfg.keys())
    cfg_str = ','.join([f'{k}:{cfg[k]}' for k in ks])
    return cfg_str.lower()

def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write(f'{cfg_str}\n')

def run(cfg_file, config_idx, dry_run=False):
    """ Sweep the config file and only pick the one at config_idx """
    sweeper = Sweeper(cfg_file)
    configs = sweeper.config_dicts
    cfg = sweeper.generate_config_for_idx(config_idx)
    for k in ['num_combinations','config_idx']:
        del cfg[k]
    outdir = './configs'
    used_cfg_file = f'{outdir}/used_configs.txt'

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    save_used_cfg(cfg, used_cfg_file)

    flags = load_args(cfg)

    print('------------------------------')
    print(f'Run: {config_idx}')
    print('------------------------------')
    print('\n'.join([f'{k}: {v}' for k,v in cfg.items() if len(configs[k])>1]))
    print(f'Flags:\n{flags}')

    if not dry_run:
        call(f'python mbexp.py {flags}', shell=True)

def load_args(cfg):
    """ keys: env, alg, seed_train """
    # alg": ["POPLINA", "POPLINP", "POPLINAE", "POPLINPE", "PETS", "DecentCEM"],
    alg = cfg['alg']
    env = cfg['env']
    seed = cfg['seed_train']
    ntrain_iters = {"gym_hopper": 100, "gym_reacher": 200, "gym_walker2d": 100,
            "gym_cheetah": 100, "gym_fswimmer": 50, "pusher": 200,
            "gym_cartpole": 250, "reacher": 100, "gym_invertedPendulum": 50,
            "gym_pendulum": 50, "gym_acrobot": 50, "gym_fhopper": 100,
            "gym_ant":100, "halfcheetah":50}
    task_hor = {"gym_hopper": 1000, "gym_reacher": 50, "gym_walker2d": 1000,
            "gym_cheetah": 1000, "gym_fswimmer": 1000, "pusher":150,
            "gym_cartpole": 200, "reacher": 150, "gym_invertedPendulum":100,
            "gym_pendulum": 200, "gym_acrobot": 200, "gym_fhopper": 1000,
            "gym_ant":1000, "halfcheetah":1000}
    var_args = f'-env {env} -logdir log/{env}/{alg} \
            -o exp_cfg.exp_cfg.ntrain_iters {ntrain_iters[env]} \
            -o exp_cfg.sim_cfg.task_hor {task_hor[env]} \
            -o exp_cfg.sim_cfg.seed_train {seed} '
    fixed_args = load_args_template(alg)
    return var_args + fixed_args

def load_args_template(alg):
    with open(f'configs/{alg}.json', 'r') as f:
        args_dict = json.load(f)
    return ' '.join([f'{k} {v}' for k, v in args_dict.items()])

if __name__ == "__main__":
    print('Note this is for running param search on compute canada')
    if len(sys.argv) < 3:
        print('Usage: python param_search_cc.py <config file> <config_idx>')
    else:
        run(sys.argv[1], int(sys.argv[2]))

