import os
import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.datasets.inaturalist import make_inaturalist

from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# --
log_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)  # Limit PyTorch to a single thread

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, max_batches=None):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = True  # Always load a model for evaluation
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    device = torch.device('cpu')  # Use CPU instead of GPU

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = min(args['data']['num_workers'], 6)  # Reduce to 6 or less
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loader
    _, unsupervised_loader, unsupervised_sampler = make_inaturalist(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=False,
        copy_data=copy_data,
        drop_last=True)

    ipe = len(unsupervised_loader)

    # -- load pre-trained checkpoint
    encoder, predictor, target_encoder, _, _, _ = load_checkpoint(
        device=device,
        r_path=load_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        opt=None,
        scaler=None)

    if torch.cuda.is_available():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    else:
        encoder = encoder
        predictor = predictor
        target_encoder = target_encoder

    for p in target_encoder.parameters():
        p.requires_grad = False

    results = []

    logger.info('Running evaluation')

    loss_meter = AverageMeter()
    maskA_meter = AverageMeter()
    maskB_meter = AverageMeter()
    time_meter = AverageMeter()

    iteration_count = 0

    with torch.no_grad():  # Ensure entire evaluation loop is within no_grad context
        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            if iteration_count >= 10:  # Stop after 10 iterations
                break

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)

            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def eval_step():
                # --

                def forward_target():
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                    B = len(h)
                    # -- create targets (masked regions of h)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                    return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)
                else:
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                return (float(loss))

            loss, etime = gpu_timer(eval_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Collect results
            results.append({
                'itr': itr,
                'loss': loss,
                'maskA': maskA_meter.val,
                'maskB': maskB_meter.val,
                'time': etime,
            })

            # -- Logging
            def log_stats():
                csv_logger.log(1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   torch.cuda.max_memory_allocated() / 1024. ** 2,
                                   time_meter.avg))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

            iteration_count += 1

    unsupervised_loader = None

    # Save results to file
    results_file = os.path.join(folder, 'evaluation_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()


