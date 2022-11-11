import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.config.config import OPTION as opt
from libs.dataset.transform import TestTransform, TrainTransform
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.models.HPAN.HPAN import HPAN
from libs.utils.Logger import Logger, Loss_record, TimeRecord
from libs.utils.Logger import TreeEvaluation as Evaluation
from libs.utils.loss import cross_entropy_loss, mask_iou_loss, proto_dist_loss
from libs.utils.optimer import PANNet_optimizer
from libs.utils.Restore import get_save_dir, restore, save_model
from tqdm import tqdm
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')

    # 文件路径设置
    parser.add_argument("--snapshot_dir", type=str, default=opt.SNAPSHOTS_DIR)
    parser.add_argument("--data_path", type=str, default=opt.DATASETS_DIR)
    parser.add_argument("--arch", type=str, default='HPAN')
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=4)

    # 载入模型参数
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--save_epoch", type=int, default=5)

    # 输入视频参数
    parser.add_argument("--input_size", type=int, default=opt.TRAIN_SIZE)
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)
    parser.add_argument("--sample_per_class", type=int, default=100)

    # GPU显存相关参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 训练时长相关参数
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--step_iter", type=int, default=30)
    parser.add_argument("--max_epoch", type=int, default=101)

    # 验证相关参数
    parser.add_argument("--novalid", action='store_true')

    # 网络参数
    parser.add_argument("--backbone_type", type=str, default='resnet50')
    parser.add_argument("--with_prior_mask", action='store_true')
    parser.add_argument("--with_proto_attn", action='store_true')
    parser.add_argument("--proto_with_self_attn", action='store_true')
    parser.add_argument("--proto_per_frame", type=int, default=5)
    parser.add_argument("--with_domain_attn", action='store_true')
    parser.add_argument("--domain_with_self_attn", action='store_true')
    parser.add_argument("--with_prior_mask_loss", action='store_true')
    parser.add_argument("--with_proto_loss", action='store_true')
    parser.add_argument("--proto_loss_weight", type=float, default=1.0)

    return parser.parse_args()


def mask_criterion(pred, target):
    return [cross_entropy_loss(pred, target), mask_iou_loss(pred, target)]


# 验证
def valid_epoch(model: nn.Module, data_loader: DataLoader,
                device: torch.device, valid_evaluation: Evaluation):
    model.eval()
    for (query_img, query_mask, support_img, support_mask, query_info,
         _) in tqdm(data_loader):
        class_id = query_info['class_id']
        query_img = query_img.to(device)
        query_mask = query_mask.to(device)
        support_img = support_img.to(device)
        support_mask = support_mask.to(device)
        with torch.no_grad():
            pred_map = model(query_img, support_img, support_mask)
        pred_map = pred_map.squeeze(2)
        query_mask = query_mask.squeeze(2)
        valid_evaluation.update_evl(class_id, query_mask, pred_map)


def train(args, logger: Logger):
    # build models
    logger.info('Building model...')

    net = HPAN(backbone_type=args.backbone_type,
               with_prior_mask=args.with_prior_mask,
               with_proto_attn=args.with_proto_attn,
               proto_with_self_attn=args.proto_with_self_attn,
               proto_per_frame=args.proto_per_frame,
               with_domain_attn=args.with_domain_attn,
               domain_with_self_attn=args.domain_with_self_attn)

    total_params = sum(p.numel() for p in net.parameters())
    logger.info('Total number of parameters: {}M'.format(total_params / 1e6))

    optimizer = PANNet_optimizer(net, lr=args.lr)
    net = net.cuda()

    if args.restore_epoch > 0:
        restore(args, net)
        logger.info('Restore model from epoch {}'.format(args.restore_epoch))

    logger.info('Loading data...')
    tsfm_train = TrainTransform(args.input_size)
    tsfm_val = TestTransform(args.input_size)

    # dataloader iteration
    query_frame = args.query_frame
    support_frame = args.support_frame
    traindataset = YTVOSDataset(data_path=args.data_path,
                                train=True,
                                fold_idx=args.group,
                                query_num=query_frame,
                                support_num=support_frame,
                                sample_per_class=args.sample_per_class,
                                transforms=tsfm_train)
    validdataset = YTVOSDataset(data_path=args.data_path,
                                valid=True,
                                fold_idx=args.group,
                                query_num=query_frame,
                                support_num=support_frame,
                                sample_per_class=args.sample_per_class,
                                transforms=tsfm_val)
    # train_list = traindataset.get_class_list()
    valid_list = validdataset.get_class_list()

    train_loader = DataLoader(traindataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(validdataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            drop_last=True)

    # set loss
    logger.info('Setting loss...')

    loss_record = Loss_record()
    valid_eval = Evaluation(class_list=valid_list)

    # set epoch
    start_epoch = args.restore_epoch
    train_iters = len(train_loader)
    logger.info('Training {} epochs with {} iters per epoch'.format(
        args.max_epoch, train_iters))
    best_iou = 0
    best_f = 0
    best_j = 0
    train_time_record = TimeRecord(max_epoch=args.max_epoch,
                                   max_iter=train_iters)
    for epoch in range(start_epoch, args.max_epoch):
        logger.info('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
        # train
        net.train()

        for iter, data in enumerate(train_loader):
            query_frames, query_masks, support_frames, support_masks, _, _ = data
            # B N C H W
            query_frames = query_frames.cuda()
            query_masks = query_masks.cuda()
            support_frames = support_frames.cuda()
            support_masks = support_masks.cuda()

            result = net(query_frames,
                         support_frames,
                         support_masks,
                         with_proto_loss=args.with_proto_loss)

            if args.with_proto_loss:
                predict_masks, proto_token = result
            else:
                predict_masks = result

            predict_masks = predict_masks.squeeze(2)
            query_masks = query_masks.squeeze(2)
            ce_loss, iou_loss = mask_criterion(predict_masks, query_masks)
            total_loss = 5 * ce_loss + iou_loss
            loss_dict = {
                'total_loss': total_loss,
                'ce_loss': ce_loss,
                'iou_loss': iou_loss
            }

            if args.with_proto_loss:
                proto_loss = proto_dist_loss(proto_token)
                total_loss += proto_loss * args.proto_loss_weight
                loss_dict['proto_loss'] = proto_loss
                loss_dict['total_loss'] = total_loss

            loss_record.updateloss(loss_dict)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if iter % args.step_iter == 1 and iter > 0:
                total_time_str, remain_time_str = train_time_record.get_time(
                    epoch, iter)
                loss_str = loss_record.getloss()
                logger.info(
                    '[Train:{:03d}/{:03d}] Step: {:03d}/{:03d} Time: {}/{} | {}'
                    .format(epoch, args.max_epoch - 1, iter, train_iters,
                            total_time_str, remain_time_str, loss_str))

        is_best = False
        # validation
        if not args.novalid:
            valid_epoch(net, val_loader, torch.device('cuda'), valid_eval)
            mean_f = np.mean(valid_eval.f_score)
            str_mean_f = 'F: %.4f ' % (mean_f)
            mean_j = np.mean(valid_eval.j_score)
            str_mean_j = 'J: %.4f ' % (mean_j)
            f_list = ['%.4f' % n for n in valid_eval.f_score]
            str_f_list = ' '.join(f_list)
            j_list = ['%.4f' % n for n in valid_eval.j_score]
            str_j_list = ' '.join(j_list)
            logger.info('valid eval:')
            logger.info('{} {}'.format(str_mean_f, str_f_list))
            logger.info('{} {}'.format(str_mean_j, str_j_list))
            eval_str, eval_list = valid_eval.get_eval()

            mean_iou = eval_list[0]
            mean_f = eval_list[1]
            mean_j = eval_list[2]
            if best_iou < mean_iou:
                is_best = True
                best_iou = mean_iou
            if best_f < mean_f:
                is_best = True
                best_f = mean_f
            if best_j < mean_j:
                is_best = True
                best_j = mean_j
            logger.info('[Valid:{}/{}] | Eval: {} | Best: {}'.format(
                epoch, args.max_epoch - 1, eval_str, is_best))

        save_model(args, epoch, net, optimizer, is_best)

        if is_best:
            last_best = 0
        else:
            last_best += 1
        if last_best == 10:
            break


if __name__ == '__main__':

    # 读取参数
    args = get_arguments()

    # 创建快照文件夹
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)

    # 创建日志文件
    count = 1
    log_file = os.path.join(args.snapshot_dir,
                            'train_log_{}.txt'.format(count))
    while os.path.exists(log_file):
        count += 1
        log_file = os.path.join(args.snapshot_dir,
                                'train_log_{}.txt'.format(count))
    print('log file: {}'.format(log_file))
    logger = Logger(log_file)
    logger.info('Running parameters:')
    logger.info(str(args))

    # 训练模型
    train(args, logger)
