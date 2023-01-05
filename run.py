import os
import sys
import argparse

from utils.submitter import Submitter

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main(argv):
    """
        Examples:
                python run.py --job_name cem --script_path
                ./sbatch_param_search.sh
    """
    parser = argparse.ArgumentParser(description="Config file")
    parser.add_argument('--user', type=str, default='vincent', help='user name')
    parser.add_argument('--project_dir',
            type=str,default='/scratch/vincent/keep/decentcem', help='project directory')
    parser.add_argument('--job_name', type=str, default='cem', help='job name')
    parser.add_argument('--script_path', type=str, default='./sbatch_param_search.sh', help='sbatch script path')
    parser.add_argument('--check_time_interval', type=int, default=10, help='check time interval in minutes')
    parser.add_argument('--num_jobs', type=int, default=1, help='total num of jobs')
    args = parser.parse_args()

    cfg = dict()
    # User name
    cfg['user'] = args.user
    # Project directory
    cfg['project_dir'] = args.project_dir
    # Sbatch script path
    cfg['script_path'] = args.script_path
    # job name
    cfg['job_name'] = args.job_name
    # Job indexes list
    num_jobs = args.num_jobs
    cfg['job_list'] = list(range(1,num_jobs+1))
    # Check time interval in minutes
    cfg['check_time_interval'] = args.check_time_interval
    # cluster_name: cluster_capacity
    # cfg['clusters'] = {'Mp2':7000, 'Cedar':7000, 'Graham': 1000, 'Beluga':1000}
    cfg['clusters'] = {'Cedar': 3000}

    make_dir('output/{}'.format(args.job_name))
    submitter = Submitter(cfg)
    submitter.submit()

if __name__=='__main__':
    main(sys.argv)
