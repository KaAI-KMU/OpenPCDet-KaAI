import argparse
import os
import tqdm
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--thresholds', type=list, default=[0.0, 0.0, 0.0], help='assign score thresholds to objects') # [Vehicle, Pedestrian, Cyclist]
    parser.add_argument('--extra_tag', type=str, default='pre_annotated_labels', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def convert_boxes_for_visualize(boxes, labels, class_names):
    new_boxes = []
    for idx, box in enumerate(boxes.tolist()):
        position = {'x':box[0], 'y':box[1], 'z':box[2]}
        scale = {'x':box[3], 'y':box[4], 'z':box[5]}
        rotation = {'x':0., 'y':0., 'z':box[6]}

        psr = {'position':position, 'scale':scale, 'rotation':rotation}
        new_box = {'psr':psr, 'obj_type':class_names[int(labels[idx])-1], 'obj_id':""}
        new_boxes.append(new_box)
    
    return new_boxes


def box_filtering_by_threshold(boxes, classes, scores, threshold_list):
    if boxes.numel() == 0:
        return torch.tensor([]).cuda().long()

    threshold_list, _ = common_utils.check_numpy_to_torch(threshold_list)
    scores, _ = common_utils.check_numpy_to_torch(scores)

    if len(threshold_list.shape) == 1:
        threshold_list = threshold_list.reshape([1, -1])
    if len(scores.shape) == 1:
        scores = scores.reshape([1, -1])
    num_threshold_type = threshold_list.shape[0]

    assert int(max(classes)) <= len(threshold_list[0])
    assert scores.shape[0] == num_threshold_type
    assert len(boxes.shape) == 2

    selected = [i for i in range(boxes.shape[0])]
    for i in range(num_threshold_type):
        num_boxes = boxes.shape[0]
        threshold = [threshold_list[i][classes[idx]-1] for idx in range(num_boxes)]
        selected = [idx for idx in selected if scores[i][idx] > threshold[idx]]

    return selected


def pre_annotation(config, model, dataloader, output_dir, thresholds):
    model.eval()

    if config.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='pre-annotation', dynamic_ncols=True)

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)

        for idx in range(batch_dict['batch_size']):
            frame_id = batch_dict['frame_id'][idx]

            boxes, labels, scores = pred_dicts[idx]['pred_boxes'], pred_dicts[idx]['pred_labels'], pred_dicts[idx]['pred_scores']
            selected = box_filtering_by_threshold(boxes=boxes, classes=labels, scores=scores, threshold_list=np.array(thresholds))
            boxes, labels, scores = boxes[selected], labels[selected], scores[selected]
            if False: # for debugging
                from visual_utils import open3d_vis_utils as V
                V.draw_scenes(points=batch_dict['points'][:,1:], ref_boxes=boxes)
                
            boxes_vis = convert_boxes_for_visualize(boxes=boxes, labels=labels, class_names=dataloader.dataset.class_names)

            output_path = output_dir / (frame_id + '.json')
            f = open(output_path, 'w')
            json.dump(boxes_vis, f, indent=4)
            if cfg.LOCAL_RANK == 0:
                progress_bar.update()


def main():
    args, cfg = parse_config()

    if True:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log.txt')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    with torch.no_grad():
        pre_annotation(config=cfg, model=model, dataloader=test_loader, output_dir=output_dir, thresholds=args.thresholds)


if __name__ == '__main__':
    main()
