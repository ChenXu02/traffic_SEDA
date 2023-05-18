import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f,Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx = load_graph_data(graph_pkl_filename)
        supervisor = DCRNNSupervisor(args.filename,adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs = supervisor.evaluate('test')
        print("MAE : {}".format(mean_score))


if __name__ == '__main__':
    for j in {'cm', 'pm'}:
        for i in range(0, 10):
            if i == 0:
                filename = 'pems08_{}_0.n'.format(j)
            else:
                filename = 'pems08_{}_0.{}'.format(j, i)
            # filename='pems08_pm_0.npy'
            print(filename)
            sys.path.append(os.getcwd())
            parser = argparse.ArgumentParser()
            parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
            parser.add_argument('--config_filename', default='config.yaml', type=str,
                                help='Config file for pretrained model.')
            parser.add_argument('--filename', default=filename, type=str,)
            args = parser.parse_args()
            run_dcrnn(args)
