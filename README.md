# DecentCEM

Official repository for the paper "A Simple Decentralized Cross-Entropy Method", Zichen Zhang and Jun Jin and Martin Jagersand and Jun Luo and Dale Schuurmans, NeurIPS 2022, [arXiv link](https://arxiv.org/abs/2212.08235)

This repo supports tensorflow 1.14, 1.15, mujoco-py 1.50, python3+
It supports both local mode and cluster mode (formerly Compute Canada, now Digital Research Alliance of Canada)

## Python env installation

### Local

Install the python packages from the requirements file. It is recommended to use a virtual environment.
```
$ pip install -r requirements.txt
```

### Cluster

The following setup is for compute canada. But in principle, it can be adapted slightly to work for clusters managed by slurm.

First load python module 
```
$ module load StdEnv/2018.3 python/3.6
```

Create and activate virtualenv 
```
$ virtualenv --no-download tf1p36
$ source tf1p36/bin/activate
```

Install

```
# install directly from compute canada wheels
$ pip install –no-index -r requirements-cc-no-index.txt 
# install wheels from PyPI
$ pip install -r requirements-cc.txt
```

## Examples scripts - local

All example scripts are in the `demo_scripts/` directory
Run them from the root directory.

### Toy problem

DecentCEM 

```
$ bash demo_scripts/DecentCEM_1dopt.sh
```

CEM-GMM

```
$ bash demo_scripts/CEMGMM_1dopt.sh
```

CEM 

```
$ bash demo_scripts/CEM_1dopt.sh
```

### Pendulum

DecentCEM-A:

```
$ bash demo_scripts/DecentCEMA_PENDULUM.sh
```

DecentCEM-P:

```
$ bash demo_scripts/DecentCEMP_PENDULUM.sh
```

DecentPETS:

```
$ bash demo_scripts/DecentPETS_PENDULUM.sh
```

### Other envs

please change the following command line arguments:

```
-env 
-logdir
-o exp_cfg.exp_cfg.ntrain_iters
-o exp_cfg.sim_cfg.task_hor
```

For example, to run algorithms on gym_reacher, please check out the following scripts as references:

```
$ bash demo_scripts/DecentCEMA_reacher.sh
$ bash demo_scripts/DecentCEMP_reacher.sh
$ bash demo_scripts/DecentPETS_reacher.sh
```

## Example scripts - cluster

To run all algorithms including baselines (PETS, POPLIN-A/P) and 3 modes of DecentCEM (with ensemble size 5), for all environments,
```
python run.py --job_name e5-all --script_path ./sbatch_param_search.sh --num_jobs 390 
```

## Configurations

```
 ├──exp_cfg                                 - Experiment script configuration.
 │    ├── sim_cfg                           - Simulation configuration.
 │    ├── exp_cfg                           - Experiment configuration.
 │    │    ├── ntrain_iters                 - Number of training iterations, where each iteration runs 1 training episode
 │    │    └── ninit_rollouts               - Number of episodes before the training loop starts, mainly for collecting data for training the model
 │              
 └── ctrl_cfg (MPC)                         - Controller configuration.
      └── cem_cfg                           - CEM config
           ├── cem_type                     - Choose the variant of algorithm to use
           │                                  ['POPLINA-INIT', (for POPLIN-A, DecentCEM-A)
           │                                   'POPLINP-SEP',  (for POPLIN-P, DecentCEM-P)
           │                                   'DecentCEM',    (for DecentPETS)  
           │                                   ]
           ├── training_scheme              - Choose the training scheme
           │                                  ['BC-AR',   (for POPLIN-A)
           │                                   'AVG-R',   (for POPLIN-P)
           │                                   'BC-AE',   (for DecentCEM-A)
           │                                   'AVG-RE',  (for DecentCEM-P)
           │                                  ]
           ├── policy_network_shape         - The shape of the policy network.
           │                                  Default is [64, 64]
           │                                  POPLIN-P, DecentCEM-P: [32] (one hidden layer)
           │── ensemble_size                - The number of CEM instances 
           ├── test_policy                  - Run test/evaluation if True
           ├── test_policy_epochs           - Episodes of testing/evaluating the policy network.
           ├── eval_ctrl_type               - "MPC" or "PolicyControl" (no MPC)
           ├── use_prev_sol                 - Use the solution from the prev time step to initialize the solution if True; otherwise use zero
           ├── debug_optimizer              - whether to debug
           ├── eval_cem_policy              - evaluate the cem policy by sampling if True; otherwise use the cost in the last iteration as the cost for the cem policy
      └── opt_cfg                           - optimization config
           ├── init_var                     - the variance of the initial sample distribution

   -ca model-type      The model that will be used: whether it's deterministic or probabilistic neural network (output mean and variance). 
                      'PE' indicates an ensemble of probabilistic networks
   -ca prop-type    : The propagation method that will be used. 'E' refers to expectation
   -ca opt-type     : The optimizer that will be used to select action sequences.
           Select from [CEM, DecentCEM, POPLIN-A, POPLIN-P, POPLIN-A-E, POPLIN-P-E].

    Please refer to the [PETS](https://github.com/kchua/handful-of-trials) repo for the full options for the arguments above. But note that only the values used above were tested in this project.

```

### Ensemble Size

To change the ensemble size, besides changing the following parameter:

```
-o ctrl_cfg.cem_cfg.ensemble_size 5
```

we also need to adjust the sample size parameters for each ensemble size, as they are specified in the config file of each environment.
For instance, `dmbrl/config/gym_acrobot.py`, for the DecentCEM-A algorithm, we would need to change the values of `popsize` and `num_elites` in the dictionary after `POPLIN-A-E`.

The default in the main branch is ensemble size of 5.
For the convenience of running the code for other ensemble sizes, we've made a separate branch for each ensemble size: E2,E3,E4,E6

The best ensemble size for each environment and algorithm is:

```
A for DecentCEM-A, P for DecentCEM-P, C for DecentPETS

gym_pendulum,           A4,P2,C4
gym_acrobot,            A3,P5,C4
gym_cartpole,           A6,P6,C5
gym_invertedPendulum,   A6,P3,C3
gym_fswimmer,           A4,P3,C2
gym_reacher,            A6,P3,C6
gym_hopper,             A2,P2,C5
gym_walker2d,           A3,P6,C5
gym_cheetah,            A6,P5,C2
gym_ant,                A5,P6,C6
reacher,                A6,P3,C2
halfcheetah,            A4,P6,C2
pusher,                 A4,P3,C2
```

## Visualizations

Run the following command from the root directory to plot the results of 1d toy example as well as the mujoco learning curves.

```
$ python utils/plotter.py
```
For the plot for 1d toy example, it expected the results described in `batch_toy.sh`
The results are stored in csv files. Refer to `utils/cem_opt_1d.py`.

For the plot for mujoco learning curves, it expected the results of all algorithms. It can be produced by following the steps in [this section](#example-scripts---cluster)
The results are stored in mat files. Refer to line 204 in the file `dmbrl/misc/MBExp.py`.

## Acknowledgments

This code base is built on top of the following repos:

[PETS](https://github.com/kchua/handful-of-trials): PETS algorithm
[POPLIN](https://github.com/WilsonWangTHU/POPLIN): POPLIN-A/P algorithms
[mbbl](https://github.com/WilsonWangTHU/mbbl): gym environments for benchmarking mbrl
[Explorer](https://github.com/qlan3/Explorer): scripts for [compute canada](https://docs.alliancecan.ca/wiki/Getting_started)


## Troubleshoot

### Tensorflow_gpu==1.15.0(or 1.14.1) does not install properly on the cluster

as of Jan 4, 2023, tensorflow_gpu does not seem to install properly,
where the following command complains about unresolved dependencies
`$ pip install --no-index tensorflow_gpu==1.15.0`

I suspect that it's due to updated packages in the cluster package repository.
If you encounter this issue, the best solution is to report to the Compute Canada support team.

### not able to install mujoco-py locally

Reproduce:  

``` pip install mujoco-py==1.50.1.68 ```

Error message:  
``` conflicting types for ‘PFNGLFRAGMENTLIGHTFVSGIXPROC’ ```

Resolve the issue:

comment out some lines in `/usr/include/GL/glext.h` following the steps here:
https://github.com/openai/mujoco-py/issues/383#issuecomment-965154631


## Citation
If you find this repo useful in your research, please cite

```
@inproceedings{
zhang2022a,
title={A Simple Decentralized Cross-Entropy Method},
author={Zichen Zhang and Jun Jin and Martin Jagersand and Jun Luo and Dale Schuurmans},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=IQIY2LASzYx}
}
```
