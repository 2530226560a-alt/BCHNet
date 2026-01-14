r""" Improved BCHNet training & val code with enhanced features """

import sys
sys.path.insert(0, "../")

import argparse

import torch.optim as optim
import torch.nn as nn
import torch, gc
import time
from tqdm import tqdm, trange
import torch.distributed as dist
from model.BCHNet_improved import ImprovedBCHNetwork
from common.logger import Logger, AverageMeter, count_params
from common.evaluation import Evaluator
from common import utils
from data.datasetDDP import FSSDataset


def train(epoch, model, dataloader, optimizer, scheduler, training):
    r""" Train Improved BCHNet """

    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    train_bar = tqdm(dataloader, file=sys.stdout, desc='Processing epoch{}'.format(epoch))
    l = len(dataloader)

    for idx, batch in enumerate(train_bar):

        # 1. BCHNetworks forward pass
        batch = utils.to_cuda(batch)

        logit_mask_1, logit_mask_2, mask_rough = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask_1.argmax(dim=1)
        
        # 2. Compute enhanced loss with multiple components
        # Primary query-to-support loss
        loss_1 = model.module.compute_objective(logit_mask_1, batch['query_mask'], logit_mask_2)
        
        # Rough mask supervision loss
        loss_rough = model.module.pred_mask_loss(mask_rough, batch['query_mask'])
        
        # Bidirectional consistency loss
        loss_2 = model.module.compute_objective(logit_mask_2, batch['support_masks'].squeeze(1))
        
        # Combined loss
        loss = loss_1 + 0.5 * loss_rough + 0.3 * loss_2
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, l, epoch, write_batch_idx=100)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Improved Cross-Domain Few-Shot Semantic Segmentation')
    parser.add_argument('--datapath', type=str, default='../datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=30)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50'])
    parser.add_argument('--load', type=str, default='False')
    parser.add_argument('--use_enhanced_filter', type=str, default='True', help='Use enhanced frequency filter')
    parser.add_argument('--use_attention', type=str, default='True', help='Use attention mechanisms')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'poly'])
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs')
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Distributed training
    import os
    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    dist.init_process_group(backend='gloo',init_method='env://',rank=0,world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Model initialization
    use_enhanced_filter = args.use_enhanced_filter == 'True'
    use_attention = args.use_attention == 'True'
    model = ImprovedBCHNetwork(args.backbone, use_enhanced_filter=use_enhanced_filter, use_attention=use_attention)
    Logger.log_params(model)

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn, sampler_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_val, sampler_val = FSSDataset.build_dataloader('fss', args.bsz, args.nworker, '0', 'val')
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    map_location = 'cuda:%d' % local_rank
    
    # Load trained model
    if args.load == 'True':
        path = './logs/test_case.log/best_model.pt'
        src_dict = torch.load(path, map_location={'cuda:0': map_location})
        model.load_state_dict(src_dict, strict=False)
        Logger.info('Loaded pretrained model from: %s' % path)

    # Optimizer with different learning rates for different components
    backbone_params = []
    attention_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'attention' in name or 'cross_attention' in name or 'pyramid' in name:
            attention_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": attention_params, "lr": args.lr * 0.5},
        {"params": backbone_params, "lr": args.lr * 0.1}
    ], weight_decay=1e-4)

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niter, eta_min=args.lr * 0.01)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:  # poly
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / args.niter) ** 0.9)

    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_lr = args.lr
    
    Logger.info('==================== Starting Training ====================')
    Logger.info('Enhanced Filter: %s, Attention Modules: %s' % (use_enhanced_filter, use_attention))
    
    for epoch in range(args.niter):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("Epoch:{}/{}  Lr:{:.2E}".format(epoch, args.niter, current_lr))

        sampler_trn.set_epoch(epoch)

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, scheduler, training=False)

        # Learning rate scheduling
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Save the best model
        if epoch == 0:
            Logger.save_model_miou(model, epoch, val_miou)
            best_val_miou = val_miou

        elif val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)
            Logger.info('New best model saved! mIoU: %.2f' % val_miou)
        
        # Also save based on loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save loss-based best model
            save_path = Logger.logpath + '/best_loss_model.pt'
            torch.save(model.state_dict(), save_path)
        
        gc.collect()
        torch.cuda.empty_cache()

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.add_scalar('data/learning_rate', current_lr, epoch)
        Logger.tbd_writer.flush()
        
    Logger.info('==================== Finished Training ====================')
    Logger.info('Best Validation mIoU: %.2f' % best_val_miou)
    Logger.tbd_writer.close()
