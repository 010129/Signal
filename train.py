from utils.logger import setup_logger
from data import make_dataloader
from modeling import make_frame
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler310 import WarmupMultiStepLR
from layers.make_loss import make_loss
from engine.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Signal Training")
    parser.add_argument(
        "--config_file", default="configs/RGBNT201/Signal.yml", help="path to config file", type=str
    )
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)


    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  

    new_output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ckpt_save_path)
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    logger = setup_logger("Signal", new_output_dir, if_train=True)
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(args.local_rank)
    else:
        gpu_info = "No GPU available"
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    logger.info("Local GPU model: {}, Torch version: {},cuda:{}".format(gpu_info, torch_version,cuda_version))
    logger.info("Saving model in the path :{}".format(new_output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if hasattr(model, 'flops'):   
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of parameters:{n_parameters / 1e6}")

    else:
        print("model has no flops")
        
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes) # ID_loss,Triplet_loss,center_loss

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.DATASETS.NAMES not in ["MSVR310"]:
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        
    # scheduler = create_scheduler(cfg, optimizer)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, 
        args.local_rank,
        cfg.MODEL.stageName
    )


