from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor




def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f,Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(args.filename,adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    for j in {'cm', 'pm'}:
        for i in range(0, 10):
            if i == 0:
                filename = 'pems08_{}_0.n'.format(j)
            else:
                filename = 'pems08_{}_0.{}'.format(j, i)
            # filename='pems08_pm_0.npy'
            print(filename)
            parser = argparse.ArgumentParser()
            parser.add_argument('--config_filename', default='config.yaml', type=str,
                                help='Configuration filename for restoring the model.')
            parser.add_argument('--filename', default=filename, type=str, )
            parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
            args = parser.parse_args()
            main(args)
