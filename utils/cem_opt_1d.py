import numpy as np   
import scipy.stats as stats
from tqdm import trange
import argparse
import os
import csv
from time import localtime, strftime, time
from datetime import datetime
import json

# add root path for local import
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# local module
from scripts.test_opt_func import hard1d1
from dmbrl.misc import logger


def cem(obj, init_mean, init_var, lb, ub, args, debug=True, csv_debug_writer=None, instance=0):
    """ Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            obj (handle of a function): obj(x) where x has shape n x d with n samples of dim d. 
                                        The goal is to minimize this function
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        
        Returns:
            sol (np.ndarray): New mean of the candidate distribution
    """
    mean, var, t = init_mean, init_var, 0
    sol_dim = mean.shape[0]
    # truncated gaussian, rejection sample from mu-2sigma to mu+2sigma
    # the original (-2 2) is due to init_var = 0.25, sigma= 0.5,
    # based on the formula on the scipy docs, 
    # the clip val in the func argument is computed by 1/sigma  = 1/0.5 = 2
    # another way  to compute the clip val is that we know that the sigma is set to be sigma=(ub-mean)/2 =ub/2
    # hence b = ub/sigma = 2. 
    # the lower clip value 'a' can be computed similarly
    #sigma = np.sqrt(init_var[0])
    #X = stats.truncnorm((lb[0]-mean[0])/sigma, (ub[0]-mean[0])/sigma, loc=np.zeros_like(mean), scale=np.ones_like(mean))
    std = np.sqrt(init_var)
    X = stats.truncnorm((lb-mean)/std, (ub-mean)/std, loc=np.zeros_like(mean), scale=np.ones_like(mean))
    return_best_sample = False
    popsize = round(float(args.popsize) / args.ensemble_size)
    num_elites = round(args.elites_ratio * popsize)

    while (t < args.max_iters) and np.max(var) > args.epsilon:
        lb_dist, ub_dist = mean - lb, ub - mean
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

        samples = X.rvs(size=[popsize, sol_dim]) * np.sqrt(constrained_var) + mean
        costs = obj(samples) # shape (n,)
        sorted_idx = np.argsort(costs) 
        elites = samples[sorted_idx][:num_elites]

        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)

        mean = args.alpha * mean + (1 - args.alpha) * new_mean
        var = args.alpha * var + (1 - args.alpha) * new_var

        cost_mean = obj(mean[np.newaxis, :])
        cost_new_mean = obj(new_mean[np.newaxis, :])

        if csv_debug_writer:
            csv_debug_writer.writerow([t,instance,mean[0],var[0],cost_mean[0],new_mean[0],new_var[0],cost_new_mean[0], samples[:,0].tolist(),costs.tolist(),sorted_idx[:num_elites].tolist()] )

        if debug:
            logger.info(f'Iter: {t}')
            logger.info("samples: {}".format(samples))
            logger.info('new mean of elite: {}'.format(new_mean))
            #logger.info('new variance of elite: {}'.format(new_var))
            logger.info('current mean of elite: {}'.format(mean))
            #logger.info('current variance of elite: {}'.format(var))
            logger.info('Mean cost of elites: {}'.format(
                np.mean(costs[sorted_idx][:num_elites]))
            )
            logger.info('Best Cost found: {}'.format(costs[sorted_idx[0]]))

        t += 1

    if return_best_sample:
        sol, solvar = elites[0], var # elites sorted from lowest cost to higher cost
    else:
        sol, solvar = mean, var

    # eval the cost of the sol
    cost = obj(sol[np.newaxis, :])

    return sol, cost[0]

def cem_gmm(obj, init_mean, init_var, lb, ub, args, debug=True, csv_debug_writer=None):
    """ CEM with a GMM sampling distribution 
    
        Reference: https://arxiv.org/pdf/1907.04202.pdf
        "Variational Inference MPC for Bayesian Model-based Reinforcement Learning"
    """
    sol_dim = init_mean.shape[0]
    return_best_sample = False
    popsize = args.popsize
    num_elites = round(args.elites_ratio * popsize)
    eps_div = 1e-6 # to prevent from dividing by zero
    
    # initialize the GMM parameters
    M = args.M # number of gaussians
    pi = np.ones((M, 1)) / M
    init_std = np.sqrt(init_var)
    gmm_var = np.array([init_var] * M)
    if args.randomize_init_mean == 1:
        # uniform dist for random restart
        gmm_mu = np.random.uniform(lb, ub, size=(M, sol_dim))
    elif args.randomize_init_mean == 2:
        # normal distribution for random restart
        X = stats.truncnorm((lb-init_mean)/init_std, (ub-init_mean)/init_std, loc=np.zeros_like(init_mean), scale=np.ones_like(init_mean))
        gmm_mu = X.rvs(size=[M, sol_dim]) * np.sqrt(init_var) + init_mean
    else:
        # fixed dist for random restart
        gmm_mu = np.array([init_mean] * M)

    t = 0
    X = stats.truncnorm((lb-init_mean)/init_std, (ub-init_mean)/init_std,
                       loc=np.zeros_like(init_mean), scale=np.ones_like(init_mean))
    kappa = args.kappa # 0.5 or 0.25
    

    while (t < args.max_iters) and np.max(gmm_var) > args.epsilon:

        # pre-compute the lb_dist and ub_dist for each mode in GMM
        # gmm_mu : [M, sol_dim]
        # gmm_var : [M, sol_dim]
        # lb, ub: sol_dim,  
        lb_dist_gmm, ub_dist_gmm = gmm_mu - lb, ub - gmm_mu
        samples = np.zeros((popsize, sol_dim)) 
        pi_cumsum = np.cumsum(pi)
            
        # first sample from standard normal, then shift and scale to match the distribution 
        std_normal_samples = X.rvs(size=(popsize, sol_dim)) 
        # sample from this given GMM defined by (pi, mu, sigma) each of m x 1
        r = np.random.uniform(0,1,size=popsize)
        mode_indices = np.searchsorted(pi_cumsum, r)  # M,
        # notation wise, the equal of r <= pi_cum_sum is better removed but it should be okay for float number 
        sorted_gmm_mu = gmm_mu[[mode_indices]]  
        sorted_gmm_var = gmm_var[[mode_indices]]  
        sorted_lb_dist_gmm = sorted_gmm_mu - lb
        sorted_ub_dist_gmm = ub - sorted_gmm_mu
        constrained_var = np.minimum(np.minimum(np.square(sorted_lb_dist_gmm / 2), np.square(sorted_ub_dist_gmm / 2)), sorted_gmm_var)
        samples = std_normal_samples * np.sqrt(constrained_var) + sorted_gmm_mu # popsize, sol_dim
            
        ## slower version, non-vectorized
        # for i_sample in range(popsize):
            #for i, th in enumerate(pi_cumsum):
            #    if r < th:
            #        m = i
            #        break
            #mean = gmm_mu[m]
            #var = gmm_var[m]
            #constrained_var = np.minimum(np.minimum(np.square(lb_dist_gmm[m] / 2, np.square(ub_dist_gmm[m] / 2)), var)
            #samples[i_sample] =  std_normal_samples * np.sqrt(constrained_var) + mean
        ## END: slower version, non-vectorized
        
        costs = obj(samples) # shape (n,)
        sorted_idx = np.argsort(costs) 
        elites = samples[sorted_idx][:num_elites]
        W = np.zeros((popsize,)) # weights as in Eq(8)
        W[sorted_idx[:num_elites]] = 1 # set the weights of elites to one
        
        # qa: prob_samples, in eq (6)
        qa = np.array([1.0 / popsize] * popsize) # simplication from Eq (7), Wj is initialized to uniform 1/K
         
        wk = W * qa **(-kappa)
        wk = wk / np.sum(wk) # wk is K by 1: weights of each samples
        
        # next, compute wmk, Nm in Eq (10)
        # version 1: non vectorized
        eta = np.zeros( (M, popsize) )
        for i in range(M):
            #if i == m:
            #    eta[m, :] = pi[m] * X.pdf(std_normal_samples)[:, 0] # K x 1
            # add probs from other unchosen gaussians
            gmm_constrained_var = np.minimum(np.minimum(np.square(lb_dist_gmm[i] / 2), np.square(ub_dist_gmm[i] / 2)), gmm_var[i])
            prob_samples_other_dist = X.pdf( (samples - gmm_mu[i]) / np.sqrt(gmm_constrained_var) )
            eta[i, :] = pi[i] * prob_samples_other_dist[:, 0]
        # # version 2: vectorized
        # gmm_constrained_var = np.minimum(np.minimum(np.square(lb_dist_gmm / 2), np.square(ub_dist_gmm / 2)), gmm_var) # M,
        # prob_samples = X.pdf( (np.tile(samples, (M, 1, 1)).transpose(1,0,2) - gmm_mu) / np.sqrt(gmm_constrained_var) ) # samples: popsize, sol_dim; tile to popsize, M, sol_dim; gmm_mu: M, sol_dim -> prob: popsize, M, sol_dim
        # eta = np.tile(pi, (1, popsize)) * np.squeeze(prob_samples).transpose(1,0) # pi: (M, 1) tile to (M, K); prob_samples: (K, M, sol_dim)->(M,K), eta: (M, K)

        # normalize
        eta = eta/np.sum(eta, axis=0) # sum over the M modes in GMM, shape: M, K, some eta(m, k) = 0 due to the truncnorm, but sum over all modes should be positive
        
        wk = np.tile(wk, (M, 1))# expand dim for subsequent operations a, some entries is zero due to indicator function in CEM
        wmk = eta * wk # M, K # most issues arise here, since the nonzero items in eta and wk do not overlap, causing 0 in all sample s in the mode
        Nm = np.sum(wmk, axis=1, keepdims=1) # M, 1
        wmk =  wmk / (Nm + eps_div) # (M, K) / M
        
        wmk_tile = np.tile(wmk[:, :, np.newaxis], (1,1,sol_dim)) # M, K, soldim
        samples_tile = np.tile(samples, (M, 1, 1))
        # update gmm params
        new_gmm_mu  = np.sum(wmk_tile * samples_tile, axis=1)
        new_gmm_var = np.sum(wmk_tile * (samples_tile - np.tile(new_gmm_mu[:, np.newaxis, :], (1,popsize,1)))**2, axis=1)
        new_gmm_var[new_gmm_var==0] = args.epsilon # set zero vars to epsilon
        new_pi = Nm / sum(Nm)

        gmm_mu = args.alpha * gmm_mu + (1 - args.alpha) * new_gmm_mu
        gmm_var = args.alpha * gmm_var + (1 - args.alpha) * new_gmm_var
        pi = args.alpha * pi + (1 - args.alpha) * new_pi

        cost_mu = obj(gmm_mu)
        cost_new_mu = obj(new_gmm_mu)

        if csv_debug_writer:
            for i in range(M): # over instances
                csv_debug_writer.writerow([t,i,pi[i][0], gmm_mu[i][0],gmm_var[i][0],cost_mu[i],new_pi[i][0], new_gmm_mu[i][0],new_gmm_var[i][0],cost_new_mu[i], samples[:,0].tolist(),costs.tolist(),sorted_idx[:num_elites].tolist()] )

        t += 1


    if args.return_mode == "d":
        # deterministic, return the mode with max prob
        max_mode = np.argmax(pi)
        # stochastic, sample the mode based on the prob
        sol = gmm_mu[max_mode]
    elif args.return_mode == "s":
        r = np.random.uniform(0,1)
        max_mode = np.searchsorted(np.cumsum(pi), r)
        sol = gmm_mu[max_mode]
    elif args.return_mode == "m":
        cost_mode = obj(gmm_mu) # M,1
        max_mode = np.argmin(cost_mode)
        sol = gmm_mu[max_mode]
    else:
        assert False, f'the return mode {args.return_mode} does not exist' 
    
    cost = obj(sol[np.newaxis, :])
    
    if csv_debug_writer:
        # write the final sol after argmax
        csv_debug_writer.writerow([-1,-1,sol,cost[0],max_mode,pi[:,0].tolist()])

    return sol, cost[0]

def cem_ensemble(obj, init_mean, init_var, lb, ub, args, debug=True, csv_debug_writer=None):
    """ 
        args.ensemble_size = 5. fallback to CEM if ensemble_size is 1
        init_var is fixed for each worker
        add random noise to init_mean for each worker
    """
    num_workers = args.ensemble_size
    if num_workers == 1:
        sol, cost = cem(obj, init_mean, init_var, lb, ub, args, debug, csv_debug_writer) 
        # write the final sol after argmax
        if csv_debug_writer:
            csv_debug_writer.writerow([-1,-1,sol,cost,0])
        return sol, cost

    sol_workers, cost_workers = [], []
    mean, var = init_mean, init_var
    sol_dim = mean.shape[0]
    best_sol = mean
    cost_min = 1e5
    idx_min = 0

    for i in trange(num_workers):
        # obtain the sol from each worker, then do argmax
        if args.randomize_init_mean == 1:
            worker_init_mean = np.random.uniform(lb, ub)
        elif args.randomize_init_mean == 2:
            # normal distribution for random restart
            sigma = np.sqrt(init_var)
            X = stats.truncnorm(lb/sigma, ub/sigma, loc=np.zeros_like(mean), scale=np.ones_like(mean))
            worker_init_mean = X.rvs(size=[sol_dim]) * np.sqrt(var) + mean
        else:
            worker_init_mean = init_mean

        sol, cost = cem(obj, worker_init_mean, init_var, lb, ub, args, False, csv_debug_writer, i)
        logger.info(f'sol of the worker {i} is {sol}')
        logger.info(f'cost of the worker {i} is {cost}')
        sol_workers.append(sol)
        cost_workers.append(cost)

        if cost < cost_min:
            # if better than previous workers
            cost_min = cost
            idx_min = i
            best_sol = sol
    
    if csv_debug_writer:
        # write the final sol after argmax
        csv_debug_writer.writerow([-1,-1,best_sol,cost_min,idx_min])
    logger.info(f'mean, min, max action of the {num_workers} cem workers: {np.mean(sol_workers,0), np.min(sol_workers, 0), np.max(sol_workers, 0)}') 
    logger.info(f'mean, min, max cost of the cem workers: {np.mean(cost_workers), np.min(cost_workers), np.max(cost_workers)}') 
    
    return best_sol, cost_min



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs')
    parser.add_argument('--alg', dest='alg', type=str, help='options: CEM, or CEM-E, or CEM-GMM')
    parser.add_argument('--max_iters', dest='max_iters', type=int, default=100, help='max number of iters of optimization')
    parser.add_argument('--popsize', dest='popsize', type=int, default=1000, help='total population size of CEM')
    parser.add_argument('--ensemble_size', dest='ensemble_size', type=int, default=1, help='ensemble size of CEM')
    parser.add_argument('--randomize_init_mean', dest='randomize_init_mean', type=int, default=0, help='whether to randomize the init mean of CEM, default is False')
    parser.add_argument('--elites_ratio', dest='elites_ratio', type=float, default=0.1, help='ratio of elites')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=1e-3, help='epsilon threshold of variance of samples')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='smoothing factor of the mean, next_mu = alpha * old_mu + (1-alpha) * elite_mean')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='random seed')
    parser.add_argument('--episodes', dest='episodes', type=int, default=10, help='number of times to run the optimizer')
    parser.add_argument('--M', dest='M', type=int, default=1, help='number of modes in GMM')
    parser.add_argument('--kappa', dest='kappa', type=float, default=0.5, help='entropy regularizer')
    parser.add_argument('--return_mode', dest='return_mode', type=str, default="s", help='options: s for stochastic, or d for deterministic, or m for max mode')

    args = parser.parse_args()

    if args.alg == 'CEM':
        postfix = ''
    elif args.alg == 'CEM-E': 
        postfix = f'-{args.ensemble_size}'
    else:
        postfix = f'-{args.M}'
    postfix = f'-{args.popsize}' + postfix

    logdir = os.path.join(
        f'./log/hard1d1/{args.alg}',
        datetime.now().strftime("%Y-%m-%d--%H-%M:%S-%f" + postfix)
    )
    # strftime("%Y-%m-%d--%H:%M:%S", localtime())+postfix)
    
    logger.set_file_handler(path=logdir, no_add_path=True)
    logger.info('Starting the experiments')
    np.random.seed(args.seed)
    logger.info(f"Using seed {args.seed} for training") 
    logger.info(f"The arguments are: {args}")
    
    csvfile = open(os.path.join(logdir, 'result.csv'), 'w')
    csv_writer = csv.writer(csvfile)
    # header: episode, sol, cost
    # last row: -1, cost_min, cost_mean, cost_max, cost_std
    csv_writer.writerow(['episode', 'sol', 'cost'])
    

    # store args in a config.json file
    with open(os.path.join(logdir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=False)

    dim = 1
    lb = -7.5
    ub = 7.5
    obj = hard1d1

    limit = (ub - lb)/2
    ac_lb = np.tile(lb, [dim])
    ac_ub = np.tile(ub, [dim])
    init_mean = (ac_lb + ac_ub) / 2
    var = (limit / 2)**2 # because 2*sigma = 7.5 => sigma = 3.75 => var=sigma**2
    init_var = np.tile(var, [dim])

    sol_arr = []
    cost_arr = []

    start = time()
    if args.alg == 'CEM' or args.alg == 'CEM-E': 
        if args.alg == 'CEM':
            args.ensemble_size = 1
        for i in range(args.episodes): 
            ## save the debug info on the sample distribution
            csvfile_debug = open(os.path.join(logdir, f'debug_run{i}.csv'), 'w')
            csv_debug_writer = csv.writer(csvfile_debug)
            csv_debug_writer.writerow(['00', args.ensemble_size, args.popsize])
            # header: iter,mean,var,new_mean,new_var,sample1,cost1,...sampleN,costN,elite_idx1,...elite_idxK
            csv_debug_writer.writerow(['iter', 'instance', 'mean', 'var','cost_mean', 'new_mean', 'new_var', 'cost_new_mean', '[sample1,...,sampleN]','[cost1,...,costN]','[elite_idx1...elite_idxK]'])
            sol, cost = cem_ensemble(obj, init_mean, init_var, ac_lb, ac_ub, args, False, csv_debug_writer=csv_debug_writer)
            csvfile_debug.close()
            sol_arr.append(sol)
            cost_arr.append(cost)
            csv_writer.writerow([i, np.round(sol,3), round(cost,3)])
    elif args.alg=='CEM-GMM':
        for i in range(args.episodes): 
            ## save the debug info on the sample distribution
            csvfile_debug = open(os.path.join(logdir, f'debug_run{i}.csv'), 'w')
            csv_debug_writer = csv.writer(csvfile_debug)
            csv_debug_writer.writerow(['00', args.M, args.popsize])
            # header: iter,mean,var,new_mean,new_var,sample1,cost1,...sampleN,costN,elite_idx1,...elite_idxK
            csv_debug_writer.writerow(['iter', 'instance', 'pi','mean', 'var','cost_mean', 'new_pi','new_mean', 'new_var', 'cost_new_mean', '[sample1,...,sampleN]','[cost1,...,costN]','[elite_idx1...elite_idxK]'])
            sol, cost = cem_gmm(obj, init_mean, init_var, ac_lb, ac_ub, args, False, csv_debug_writer=csv_debug_writer)
            sol_arr.append(sol)
            cost_arr.append(cost)
            csv_writer.writerow([i, sol, cost])
    else:
        logger.error(f'alg {args.alg} is not found')
        assert False, f'alg {args.alg} is not found'
    elapsed_time = time()-start
    logger.info(f'time is {elapsed_time} seconds')
    csv_writer.writerow(['time', elapsed_time])

    true_min_cost = -1.8996
    logger.info(f'Ground truth global optimum is {true_min_cost:.3f} @ 5.1457')
    logger.info(f'Stats of {args.alg} results out of {args.episodes} episodes')
    logger.info(f'min, mean, max of final solution: {np.min(sol_arr)}, {np.mean(sol_arr)}, {np.max(sol_arr)}')
    logger.info(f'min, mean, max of the cost is {np.min(cost_arr):.3f}, {np.mean(cost_arr):.3f}, {np.max(cost_arr):.3f}')
    logger.info(f'std of the cost is {np.std(cost_arr):.3f}')

    # last row: -1, cost_min, cost_mean, cost_max, cost_std
    csv_writer.writerow([-1, 'cost_min', 'cost_mean', 'cost_max', 'cost_std'])
    csv_writer.writerow([-1, round(np.min(cost_arr),3), round(np.mean(cost_arr),3), 
                        round(np.max(cost_arr),3), round(np.std(cost_arr),3)])
    csvfile.close()
