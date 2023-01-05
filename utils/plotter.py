""" This files has functions for plotting the 1d results and the learning curves for mujoco """

import glob
import json
import os
import csv 
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import uniform_filter1d as filter1d
import matplotlib.pyplot as plt


NTRAIN_ITER = {"gym_hopper": 100, "gym_reacher": 200, "gym_walker2d": 100,
            "gym_cheetah": 100, "gym_fswimmer": 50, "pusher": 200,
            "gym_cartpole": 250, "reacher": 100, "gym_invertedPendulum": 50,
            "gym_pendulum": 50, "gym_acrobot": 50, "gym_fhopper": 100,
            "halfcheetah": 50, "gym_ant": 100} # episodes
TASK_HOR = {"gym_hopper": 1000, "gym_reacher": 50, "gym_walker2d": 1000,
            "gym_cheetah": 1000, "gym_fswimmer": 1000, "pusher":150,
            "gym_cartpole": 200, "reacher": 150, "gym_invertedPendulum":100,
            "gym_pendulum": 200, "gym_acrobot": 200, "gym_fhopper": 1000,
            "halfcheetah": 1000, "gym_ant": 1000} # horizon
TITLES = {"gym_ant":"Ant", "gym_hopper": "Hopper","gym_reacher":"Reacher", "gym_walker2d": "Walker2D",
            "gym_cheetah":"HalfCheetah", "gym_fswimmer":"FixedSwimmer", "pusher":"PETS-Pusher",
            "gym_cartpole":"Cartpole", "reacher":"PETS-Reacher3D", "gym_invertedPendulum":"InvertedPendulum",
            "gym_acrobot":"Acrobot", "halfcheetah":"PETS-HalfCheetah", "gym_pendulum": "Pendulum",
            "gym_fhopper": "FixedHopper"}
COLOURS = {"PETS":"#834e56", "DecentCEM":"#f7a325", "POPLINA":"#2f9e44","POPLINP":"#9932cc",
        "POPLINAE": "#009acd", "POPLINPE": "#ff2500","SAC": "#474c4d"}
LABELS = {"PETS":"PETS", "DecentCEM": "DecentPETS", "POPLINA":"POPLIN-A", "POPLINP":"POPLIN-P",
        "POPLINAE": "DecentCEM-A", "POPLINPE":"DecentCEM-P", "SAC":"SAC"}
YLIM = {"reacher": [-200, 0], "gym_cartpole": [190, 202], "pusher": [-220, -60], "gym_walker2d": [-3500, 100]}


def plot_1d1(root_path, env):
    """  plot the results on the toy example. Reproduce fig.3 (right) in the paper """
    algs = ["CEM", "CEM-GMM", "CEM-E"]
    algs_name = ["CEM", "CEM-GMM", "DecentCEM"]

    color = ["#2f9e44", "#009acd", "#9932cc","#ff6eb4", "#ff2500"]
    popsize_list = [100, 200, 500, 1000]

    fig, ax = plt.subplots()

    def loadcsv(fname):
        with open(fname, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'time':
                    time = row[1]
                elif row[0] == '-1' and row[1] != 'cost_min':
                    cost_min, cost_mean, cost_max, cost_std = row[1:]
        return float(time), float(cost_min), float(cost_mean), float(cost_max), float(cost_std)

    for i, alg in enumerate(algs):
        cost_mean_curve = []
        cost_min_curve = []
        cost_max_curve = []
        cost_std_curve = []
        legend_label = algs_name[i] 

        for popsize in popsize_list:
            # each iter corresponds to one algorithm
            # find the csv file for each algorithm
            paths = [path for path in glob.glob(f'{root_path}/{alg}/pop{popsize}/*/result.csv')]
            print(f'{len(paths)} paths exist for {env}/{alg}: {paths}')
            # paths contain all hyper-params. 
            # we need to read the result and then return the best one for each alg
            if len(paths) == 0:
                continue
        
            time_arr = []  
            cost_min_arr = []
            cost_mean_arr = []
            cost_max_arr = []
            cost_std_arr = []

            for name in paths:
                popsize_from_path, extra_folder_name = name.strip().split('/')[-3:-1]
                extra_size = int(extra_folder_name.split('-')[-1])
                time, cost_min, cost_mean, cost_max, cost_std = loadcsv(name) 
                
                # store all the info into a list for this alg+popsize
                time_arr.append(time)
                cost_min_arr.append(cost_min)
                cost_mean_arr.append(cost_mean)
                cost_max_arr.append(cost_max)
                cost_std_arr.append(cost_std)

            # the best param has the better mean cost or same mean cost but smaller variance
            best_param_index = np.where(cost_mean_arr == np.min(cost_mean_arr))[0]
            if len(best_param_index) > 1: 
                best_std_idx = np.argmin(np.array(cost_std_arr)[best_param_index])
                best_param_index = best_param_index[best_std_idx]
            else:
                best_param_index = best_param_index[0]
            cost_min = cost_min_arr[best_param_index]
            cost_mean = cost_mean_arr[best_param_index]
            cost_max = cost_max_arr[best_param_index]
            cost_std = cost_std_arr[best_param_index]
            best_path = paths[best_param_index]
            best_time = time_arr[best_param_index]

            print(f'path for the best hyperparam under popsize {popsize} is {best_path}')
            best_config_fname = os.path.join(best_path.rsplit('/', 1)[0], 'config.json')
            with open(best_config_fname, 'r') as f:
                jdata = f.read()
                jobj = json.loads(jdata)
                print(f'best config from file {best_config_fname} is {jobj}')
            print(f'elapsed time is {best_time} seconds')
            
            cost_mean_curve.append(cost_mean)
            cost_min_curve.append(cost_min)
            cost_max_curve.append(cost_max)
            cost_std_curve.append(cost_std)

        cost_mean_curve = np.array(cost_mean_curve)
        cost_std_curve = np.array(cost_std_curve)
        if i == 1:
            ax.plot(popsize_list, cost_mean_curve, '-x', label=legend_label, color=color[i], alpha=0.5)
        else:
            ax.plot(popsize_list, cost_mean_curve, '-o', label=legend_label, color=color[i], alpha=0.5)
        ax.fill_between(popsize_list, cost_max_curve, cost_min_curve, color=color[i], alpha=0.1)
     
    font_size = 14
    ax.set_xticks( popsize_list )  # Set label locations.
    ax.set_ylim([-1.95, -1.7])
    # ax.tick_params(axis='x', labelsize=12)
    #ax.set_xscale("log")
    #ax.set_xticklabels([str(p) for p in popsize_list])
    ax.set_xlabel('Population Size', fontsize=font_size)
    ax.set_ylabel(r'Cost $f(\hat{x})$', fontsize=font_size)
    ax.legend(fontsize=font_size-2, loc=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks(list(ax.get_yticks()) + [-1.728])

    plt.grid(linestyle='--')

    text_color = 'k'
    ax.annotate(r'local optimum f(x)=-1.728@x=-2.296', xy=(200, -1.728), 
        xytext=(200+10, -1.728+0.01), color=text_color,fontsize=font_size )
    ax.annotate(r'global optimum f(x)=-1.9@x=5.146', xy=(200, -1.9), 
        xytext=(200+10, -1.9-0.02), color=text_color,fontsize=font_size )
    
    # plt.title('Comparison of CEM, CEM-GMM and DecentCEM', fontsize=font_size)
    plt.show()
    #plt.savefig(f'{root_path}/comp_1d1.png', bbox_inches='tight', dpi=600)
    #plt.savefig(f'{root_path}/comp_1d1.pdf', bbox_inches='tight', dpi=600)

def plot_lc(root_path, out_dir, envs, algs, smooth=False, legend=False, ylabel=False):
    """  plot the evaluation learning curve """
    font_size = 20
    if not os.path.exists(f'{root_path}/{out_dir}'):
        os.makedirs(f'{root_path}/{out_dir}')
    # iterate through a list of environment
    for env in envs:
        # prepare the plots, one plot for each env
        fig, ax = plt.subplots()
        num_episodes = NTRAIN_ITER[env]
        for alg in algs:
            # iterate through a list of algorithms 
            # find the mat files for each algorithm (excluding model.mat file since those are model weights)
            paths = [path for path in glob.glob(f'{root_path}/log/{env}/{alg}/*/*.mat') if "model.mat" not in path]
            print(f'path for {env}/{alg} is: {paths}')
            if len(paths) == 0:
                continue
            legend_label = LABELS[alg] 
            test_returns_list = []
            for name in paths:
                print(f"processing {name}")
                mat_data = loadmat(name)
                steps = mat_data['num_steps'] # a list with each entry denoting the steps @ which the evaluation is performed
                if steps.shape[-1] == 5: # fix a bug in the data file in sac_hopper
                    steps = np.transpose(steps, [1, 0])
                test_returns = mat_data['test_returns']

                if steps.shape[-1] > num_episodes:
                        steps = steps[:, :num_episodes]
                        test_returns = test_returns[:, :num_episodes]
                        
                test_returns_list.append(test_returns)

            num_runs = len(test_returns_list)
            test_returns = np.squeeze(test_returns_list)
            returns_mean = np.mean(test_returns, axis=0)
            returns_std_err = np.std(test_returns, axis=0) / np.sqrt(num_runs)
            x = np.cumsum(steps[0]) # skip the init_rollouts since the test/eval only starts after
            
            if smooth: 
                if env in ["gym_cartpole"]:
                    returns_mean = filter1d(returns_mean, size=30)
                else:
                    returns_mean = filter1d(returns_mean, size=10)

                ax.plot(x, returns_mean, label=legend_label, color=COLOURS[alg])
                ax.fill_between(x, returns_mean + returns_std_err, returns_mean - returns_std_err, color=COLOURS[alg], alpha=0.2)

        ax.set_xlabel('Steps', fontsize=font_size)
        if env == "gym_acrobot":
            ax.set_yticks([-400, -300, -200, -100, 0, 50])
        if ylabel:
            ax.set_ylabel('Average Return', fontsize=font_size)
        if legend:
            ax.legend()
        if YLIM.get(env, None) is not None:
            ax.set_ylim(YLIM[env])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.grid(linestyle='--')
        plt.title('{}'.format(TITLES[env]), fontsize=font_size)

        if smooth:
            if legend:
                if ylabel:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_legend_smooth.pdf', bbox_inches='tight', dpi=600)
                else:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_legend_smooth_noylabel.pdf', bbox_inches='tight', dpi=600)
            else:
                if ylabel:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_smooth.pdf', bbox_inches='tight', dpi=600)
                else:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_smooth_noylabel.pdf', bbox_inches='tight', dpi=600)
        else:
            if legend:
                if ylabel:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_legend.pdf', bbox_inches='tight', dpi=600)
                else:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_legend_noylabel.pdf', bbox_inches='tight', dpi=600)
            else:
                if ylabel:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}.pdf', bbox_inches='tight', dpi=600)
                else:
                    plt.savefig(f'{root_path}/{out_dir}/{env}_{font_size}_noylabel.pdf', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    plot_1d1("./log/hard1d1", "hard1d1")
    envs = ["gym_ant", "gym_hopper","gym_reacher", "gym_walker2d",
            "gym_cheetah", "gym_fswimmer", "pusher",
            "gym_cartpole", "reacher", "gym_invertedPendulum",
            "gym_acrobot", "halfcheetah", "gym_pendulum"]
    algs = ["PETS", "DecentCEM", "POPLINA", "POPLINP", "POPLINAE", "POPLINPE", "SAC"]
    plot_lc(".", "demo", envs, algs, True, True, False)