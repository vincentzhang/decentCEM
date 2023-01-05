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
    print('\n'.join([f'--{k}: {v}' for k,v in cfg.items() if len(configs[k])>1]))
    print(f'Flags:\n{flags}')

    if not dry_run:
        call(f'python utils/cem_opt_1d.py {flags}', shell=True)

def load_args(cfg):
    """ keys: env, alg, seed_train """
    args = ' '.join([f'--{k} {v}' for k, v in cfg.items()])
    return args

def load_args_template(alg):
    with open(f'configs/toy/{alg}.json', 'r') as f:
        args_dict = json.load(f)
    return ' '.join([f'{k} {v}' for k, v in args_dict.items()])

def run_toy(cfg_file, num_combinations):
    for i in range(1, num_combinations+1):
        print(f'running config idx {i}')
        run(cfg_file, i, dry_run=False)

if __name__ == "__main__":
    print('Note this is for running param search on local computer')
    if len(sys.argv) < 3:
        print('Usage: python param_search_toy.py <config file> <config_idx>')
    else:
        run_toy(sys.argv[-2], int(sys.argv[-1]))
        # run(sys.argv[1], int(sys.argv[2]))

