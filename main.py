import argparse
import multiprocessing as mp
import pprint
import yaml
import os
import logging
from src.utils.distributed import init_distributed
from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Print instructions to run scp manually
    print("Training completed. To download the results, run the following command:")
    print("scp your_username@remote_server_ip:/path/to/output_results/* /path/to/local_directory/")

