r""" Improved BCHNet finetuning code with enhanced features """

import argparse
import torch.distributed as dist
import torch.nn as nn
import torch
import os
from model.BCHNet_improved import ImprovedBCHNetwork
from common.logger import Logger, AverageMeter, count_params
from common.evaluation import Evaluator
from common import utils
from data.datasetDDP import FSSDataset
from common.vis import Visualizer
import torch.optim as optim
import torch.nn.functional as F
import time
from datetime import datetime


def test(model, dataloader, nshot):
    r""" Test with Self-Finetuning """

    # Define parameters to unfreeze for finetuning
    to_unfreeze_dict = [
        'module.FDF.mask_amplitude',
        'module.FDF.mask_phase',
        'module.FDF.amp_attn.0.weight',
        'module.FDF.phase_attn.0.weight',
        'module.FDF.band_weights',
    ]

    i=0
    for (name,param) in model.named_parameters():
        if any(unfreeze_name in name for unfreeze_name in to_unfreeze_dict):
            i+=1
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"Unfrozen {i} parameter groups for finetuning")

    # Compute params
    learnable_params, total_params = count_params(model)
    learnable_params = learnable_params / 1e6
    total_params = total_params / 1e6
    Logger.info('# Learnable params: %.2f M' % learnable_params)
    Logger.info('#     Total params: %.2f M' % total_params)
    
    optimizer = optim.AdamW([{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}], weight_decay=1e-5)

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        # 1. Forward pass
        batch = utils.to_cuda(batch)
        pred_mask, logit_mask_orig = model.module.predict_mask_nshot_support(batch, nshot)
        assert pred_mask.size() == batch['query_mask'].size()
        
        # 2. Finetuning step
        optimizer.zero_grad()
        loss = model.module.compute_objective_finetuning(logit_mask_orig, batch['support_masks'].clone(), nshot) 
        
        loss.requires_grad_(True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
        
        optimizer.step()

        # 3. Evaluate prediction
        with torch.no_grad():
            pred_mask_final, _ = model.module.final_predict_mask_nshot(batch, nshot)
            assert pred_mask_final.size() == batch['query_mask'].size()
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_final.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=100)
        
        # 4. Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask_final, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Improved Cross-Domain Few-Shot Semantic Segmentation Testing')
    parser.add_argument('--datapath', type=str, default='../datasets')
    parser.add_argument('--benchmark', type=str, default='deepglobe', choices=['deepglobe', 'fss', 'isic', 'lung', 'verse2D_axial', 'verse2D_sagittal'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='./logs/_0903_144122.log/12best_model.pt')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--use_enhanced_filter', type=str, default='True', help='Use enhanced frequency filter')
    parser.add_argument('--use_attention', type=str, default='True', help='Use attention mechanisms')

    # lr needs to be adjusted according to the dataset
    # 1e-7: deepglobe
    # 1e-2: fss
    # 1e-1: isic & lung & verse2D_axial & verse2D_sagittal

    parser.add_argument('--finetuning', type=str, default='True')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Distributed training
    import os
    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    dist.init_process_group(backend='gloo',init_method='env://',rank=0,world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Model initialization & load
    use_enhanced_filter = args.use_enhanced_filter == 'True'
    use_attention = args.use_attention == 'True'
    model = ImprovedBCHNetwork(args.backbone, use_enhanced_filter=use_enhanced_filter, use_attention=use_attention)
    
    Logger.info('Enhanced Filter: %s, Attention Modules: %s' % (use_enhanced_filter, use_attention))
    
    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.benchmark, args.nshot)

    # Dataset initialization & load
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test, sampler = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Load model weights
    path = args.load
    src_dict = torch.load(path, map_location='cuda:0')
    model.load_state_dict(src_dict, strict=False)
    Logger.info('Loaded model from: %s' % path)
    
    # Start time
    start_time = time.time()
    start_datetime = datetime.now().replace(microsecond=0)
    Logger.info('start_time (model loading ...): %s' % start_datetime)

    # Self Finetuning Test
    test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)

    # End time
    end_time = time.time()
    end_datetime = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time

    # Compute FPS
    FPS = len(dataloader_test)/int(total_time)
    FPS = round(FPS, 2)

    Logger.info('start_time: %s ' % start_datetime)
    Logger.info('end_time  : %s    Total time: %.2f seconds' % (end_datetime, total_time))
    Logger.info('FPS: ' + str(FPS) + ' f/s \n')

    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Self-Finetuning Testing ====================')
