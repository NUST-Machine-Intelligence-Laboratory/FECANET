r""" FECANet training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from model.FECANet import FECANet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(epoch, model, dataloader, optimizer, training, dataset):
    r""" Train FECANet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. FECANet forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1),
                           batch['support_masks'].squeeze(1), batch['history_mask'])
        pred_softmax = F.softmax(logit_mask, dim=1).detach().cpu()
        pred_mask = F.interpolate(logit_mask, size=batch['query_img'].size()[-2:],
                                  mode='bilinear', align_corners=True).argmax(dim=1)

        # update history_mask
        for j in range(batch['query_img'].shape[0]):
            sub_index = batch['idx'][j]

            # 因为dataset中都是存储的cpu格式，统一通过utils.to_cuda(batch)处理，所以必须转为tensor先
            dataset.history_mask_list[sub_index] = pred_softmax[j]  # 使用logit_mask作为pred，而不是最后的前景预测。

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='FECANet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/data/VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--loadpath', type=str, default='')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = FECANet(args.backbone, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = [0, 1], output_device=0)
    else:
        model = nn.DataParallel(model)

    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()
    current_epoch = 0
    if args.resume:
        path = args.loadpath
        checkpoint = torch.load(path)
        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.train()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn, dataset_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val, dataset_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train FECANet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(current_epoch, args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, dataset=dataset_trn, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, dataset=dataset_val, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou, optimizer)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
