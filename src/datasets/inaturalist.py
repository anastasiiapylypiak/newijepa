import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
import torchvision

logger = getLogger()

def make_inaturalist(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=6,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True
):
    dataset = INaturalist(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data
    )
    logger.info('iNaturalist dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers
    )
    logger.info('iNaturalist data loader created')
    print(f'Number of images in dataset: {len(data_loader.dataset)}')
    return dataset, data_loader, dist_sampler

class INaturalist(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder='inaturalist_dataset/',
        transform=None,
        train=True,
        copy_data=True
    ):
        data_path = root
        if copy_data:
            data_path = copy_inat_locally(root, image_folder, train)
        super(INaturalist, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized iNaturalist dataset at {}'.format(data_path))

def copy_inat_locally(
    root,
    suffix,
    image_folder='inaturalist_dataset/',
    tar_file='inaturalist_data.tar.gz',
    job_id=None,
    local_rank=None
):
    # Get the job ID from SLURM or set default
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    # Get the local rank from SLURM or set default
    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    # Define the source and target paths
    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'Source file: {source_file}\nTarget directory: {target}\nTarget file: {target_file}\nData path: {data_path}')

    # Temporary signal file to coordinate extraction among processes
    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    # Check if data has already been extracted; if not, extract it
    if not os.path.exists(data_path):
        if local_rank == 0:  # Only the master process should handle the extraction
            if not os.path.exists(tmp_sgnl_file):
                with open(tmp_sgnl_file, 'w') as f:
                    f.write('Extraction started\n')
                # Extract the tar file
                subprocess.run(['tar', '-xzf', source_file, '-C', target])
                with open(tmp_sgnl_file, 'a') as f:
                    f.write('Extraction completed\n')
            logger.info(f'Extraction completed at {data_path}')
        else:
            # Wait for the master process to complete extraction
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(10)  # Wait for 10 seconds before checking again
                with open(tmp_sgnl_file, 'r') as f:
                    if 'Extraction completed' in f.read():
                        break
            logger.info(f'Data ready at {data_path} for local_rank {local_rank}')

    return data_path