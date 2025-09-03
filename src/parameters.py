import argparse
from dataclasses import dataclass
import pandas as pd
from src.utils import multivariate as DATASETS
import json

@dataclass
class Parameters:
    data_path: str
    output_path: str
    param_path: str
    metric: str
    problem_idx: int
    problem: str
    norm: str
    itr: int
    save_distances: bool
    metric_params: dict = None
    run_type: str = "inference" # inference, loocv
    n_jobs: int = -1
    testrun: bool = False

    # Posterior init
    def __post_init__(self):
        # Check all parameters
        # Get problem name if only index is passed
        if self.problem is None:
            self.problem = DATASETS[int(self.problem_idx)]
        
    # Method that parses the arguments and initializes the Parameters object
    @staticmethod
    def parse(args: list):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--data", required=False, default="/UEA_archive/")
        parser.add_argument("-o", "--output", required=False, default="./output/")
        parser.add_argument("-x", "--problem_idx", required=False, default=0)
        parser.add_argument('-pp','--param_path',required=False,default=None)
        parser.add_argument("-p", "--problem", required=False)
        parser.add_argument("-m", "--metric", required=True)  # see regressor_tools.all_models
        parser.add_argument("-i", "--itr", required=False, default=1)
        parser.add_argument("-n", "--norm", required=False, default="none")  # none, standard, minmax
        parser.add_argument("-mp", "--metric_policy", required=False, default="inference")  # inference, loocv
        parser.add_argument('-s','--save_distances',required=False,default=None)
        parser.add_argument("-c","--metric_params",nargs='*', default=None)
        parser.add_argument("-j","--n_jobs",required=False, default=-1)
        parser.add_argument("-t", "--testrun", action="store_true", help="Flag for testing, will load fake data and generate random distances", default=False)

        arguments = parser.parse_args(args)

        data_path = arguments.data
        output_path = arguments.output
        param_path = arguments.param_path
        metric = arguments.metric
        problem_idx = arguments.problem_idx
        problem = arguments.problem
        norm = arguments.norm
        itr = arguments.itr
        metric_policy = arguments.metric_policy
        save_distances = arguments.save_distances if arguments.save_distances != 'None' else None
        n_jobs = int(arguments.n_jobs)
        testrun = arguments.testrun

        # Parse metric parameters if they are passed
        metric_params = None
        if arguments.metric_params is not None:
            metric_params = {}

            # Split on spaces
            tmp = []
            for param in arguments.metric_params:
                tmp += param.split(' ')
            arguments.metric_params = tmp

            for param in arguments.metric_params:
                key, value = param.split('=')

                # Try to convert to float, integer or leave as string
                try:
                    metric_params[key] = float(value)
                except ValueError:
                    try:
                        metric_params[key] = int(value)
                    except ValueError:
                        metric_params[key] = value

        return Parameters(data_path, output_path, param_path, metric, problem_idx, problem, norm, itr, save_distances, metric_params, metric_policy, n_jobs, testrun)

    def to_dict(self):
        return {
            'data_path': self.data_path,
            'output_path': self.output_path,
            'metric': self.metric,
            'problem_idx': self.problem_idx,
            'problem': self.problem,
            'norm': self.norm,
            'itr': self.itr,
            'save_distances': self.save_distances,
            'metric_params': self.metric_params,
            'metric_policy': self.run_type,
            'n_jobs': self.n_jobs,
            'testrun': self.testrun
        }



