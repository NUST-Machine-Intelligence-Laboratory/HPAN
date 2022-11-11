import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.config.config import OPTION as opt
from libs.dataset.transform import TestTransform
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.models.HPAN.HPAN import HPAN
from libs.utils.Logger import Logger, Loss_record, TopFiveRecord
from libs.utils.Logger import TreeEvaluation as Evaluation
from libs.utils.loss import cross_entropy_loss, mask_iou_loss
from libs.utils.optimer import finetune_optimizer
from libs.utils.Restore import get_save_dir, restore


def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')

    # 文件路径设置
    parser.add_argument("--snapshot_dir", type=str, default=opt.SNAPSHOTS_DIR)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=opt.DATASETS_DIR)
    parser.add_argument("--arch", type=str, default='HPAN')
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument("--group", type=int, default=1)

    # 载入模型参数
    parser.add_argument("--test_best", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--restore_epoch", type=int, default=0)

    # 输入视频参数
    parser.add_argument("--input_size", type=int, default=opt.TEST_SIZE)
    parser.add_argument("--query_num", type=int, default=5)
    parser.add_argument("--support_num", type=int, default=5)

    # 微调参数
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--finetune_step", type=int, default=20)
    parser.add_argument("--finetune_valstep", type=int, default=5)
    parser.add_argument("--finetune_weight", type=float, default=0.1)
    parser.add_argument("--finetune_iou", type=float, default=0.5)
    parser.add_argument("--finetune_idx", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--num_folds', type=int, default=4)

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

    return parser.parse_args()


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def criterion(pred, target):
    return [cross_entropy_loss(pred, target), mask_iou_loss(pred, target)]


def finetune(args, logger, model, imgs, masks, class_list):
    print('start finetune', args.finetune_step, args.finetune_valstep)

    if args.test_best:
        restore(args, model, test_best=True)
        logger.info('Restore best model')
    optimizer = finetune_optimizer(model, args.lr)
    loss_record = Loss_record()

    _, s_num, _, _, _ = imgs.shape
    valid_eval = Evaluation(class_list=class_list)
    model.eval()
    pred_map = model(imgs, imgs, masks)
    valid_eval.update_evl(class_list, masks.squeeze(2), pred_map.squeeze(2))
    eval_str, eval_list = valid_eval.get_eval()
    start_iou = eval_list[0]
    logger.info('Start finetune, eval: {}'.format(eval_str))

    q_img = torch.cat([imgs[:, i:i + 1] for i in range(s_num)], dim=0)
    q_mask = torch.cat([masks[:, i:i + 1] for i in range(s_num)], dim=0)
    s_img = torch.cat([
        torch.cat([imgs[:, j:j + 1] for j in range(s_num) if j != i], dim=1)
        for i in range(s_num)
    ],
                      dim=0)
    s_mask = torch.cat([
        torch.cat([masks[:, j:j + 1] for j in range(s_num) if j != i], dim=1)
        for i in range(s_num)
    ],
                       dim=0)

    model.train()
    model.apply(fix_bn)
    stop_iou = max(args.finetune_iou, start_iou)
    for train_step in range(args.finetune_step):
        pred_map, prior_mask = model(q_img,
                                     s_img,
                                     s_mask,
                                     with_prior_mask_loss=True)
        ce_loss, iou_loss = criterion(pred_map, q_mask)
        prior_ce_loss, prior_iou_loss = criterion(prior_mask, q_mask)
        total_loss = ce_loss + iou_loss + prior_ce_loss + prior_iou_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'iou_loss': iou_loss,
            'prior_ce_loss': prior_ce_loss,
            'prior_iou_loss': prior_iou_loss,
        }
        loss_record.updateloss(loss_dict)
        loss_str = loss_record.getloss()
        valid_eval.update_evl(class_list * s_num, q_mask.squeeze(2),
                              pred_map.squeeze(2))
        eval_str, eval_list = valid_eval.get_eval()

        logger.info('[Finetune: {}/{}] {} | {}'.format(train_step,
                                                       args.finetune_step,
                                                       loss_str, eval_str))
        mean_iou = eval_list[0]
        if mean_iou > stop_iou:
            break

    model.eval()
    pred_map = model(imgs, imgs, masks)
    valid_eval.update_evl(class_list, masks.squeeze(2), pred_map.squeeze(2))
    eval_str, eval_list = valid_eval.get_eval()
    stop_iou = eval_list[0]
    logger.info('Stop finetune, eval: {}'.format(eval_str))

    finetune_path = os.path.join(
        args.test_dir, 'model_fintune_{}.pth.tar'.format(class_list[0]))
    torch.save(model.state_dict(), finetune_path)

    finetune_success = True if stop_iou > start_iou else False
    return finetune_success


def test(args, logger: Logger):
    # build models
    logger.info('Building model...')

    net = HPAN(backbone_type=args.backbone_type,
               with_prior_mask=args.with_prior_mask,
               with_proto_attn=args.with_proto_attn,
               proto_with_self_attn=args.proto_with_self_attn,
               proto_per_frame=args.proto_per_frame,
               with_domain_attn=args.with_domain_attn,
               domain_with_self_attn=args.domain_with_self_attn)
    net.cuda()
    if args.test_best:
        restore(args, net, test_best=True)
        logger.info('Restore best model')

    fintune_net = HPAN(backbone_type=args.backbone_type,
                       with_prior_mask=args.with_prior_mask,
                       with_proto_attn=args.with_proto_attn,
                       proto_with_self_attn=args.proto_with_self_attn,
                       proto_per_frame=args.proto_per_frame,
                       with_domain_attn=args.with_domain_attn,
                       domain_with_self_attn=args.domain_with_self_attn)
    fintune_net.cuda()

    logger.info('Loading data...')
    tsfm_test = TestTransform(args.input_size)
    test_dataset = YTVOSDataset(data_path=args.data_path,
                                test=True,
                                fold_idx=args.group,
                                test_idx=args.finetune_idx,
                                query_num=args.query_num,
                                support_num=args.support_num,
                                transforms=tsfm_test)
    test_list = test_dataset.get_class_list()
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    logger.info('test_group: %d  test_num: %d' %
                (args.group, len(test_dataloader)))

    test_eval = Evaluation(class_list=test_list)
    finetune_eval = Evaluation(class_list=test_list)

    top5_record = TopFiveRecord()
    fintune_top5_record = TopFiveRecord()
    finetune_success = False
    while not finetune_success:
        pbar = tqdm(test_dataloader)
        for (query_img, query_mask, support_img, support_mask, query_info,
             support_info, begin_new) in pbar:
            class_id = query_info['class_id'].item()
            query_vid = query_info['video_id'].item()
            support_vids = [vid.item() for vid in support_info['video_ids']]
            top5_key = 'class_{}_vid_{}'.format(class_id, query_vid)

            if begin_new:
                s_img = support_img.cuda()
                s_mask = support_mask.cuda()
                logger.info('class_id: %d  support_vids: %s' %
                            (class_id, support_vids))
                finetune_success = finetune(args, logger, fintune_net, s_img,
                                            s_mask, test_list)
                if not finetune_success:
                    logger.info('Finetune failed, skip this support set')
                    break

            query_img = query_img.cuda()
            query_mask = query_mask.cuda()
            len_video = query_img.shape[1]
            q_num = args.query_num
            step_len = len_video // q_num
            top5_value = []
            fintune_top5_value = []
            for i in range(step_len):
                if i < step_len - 1:
                    q_img = query_img[:, i * q_num:(i + 1) * q_num]
                    q_mask = query_mask[:, i * q_num:(i + 1) * q_num]
                else:
                    q_img = query_img[:, i * q_num:]
                    q_mask = query_mask[:, i * q_num:]

                net.eval()
                with torch.no_grad():
                    p_mask = net(q_img, s_img, s_mask)
                iou = 1 - mask_iou_loss(q_mask, p_mask)
                top5_value.append(iou.data.item())
                test_eval.update_evl([class_id], q_mask.squeeze(2),
                                     p_mask.squeeze(2))

                fintune_net.eval()
                with torch.no_grad():
                    p_mask = fintune_net(q_img, s_img, s_mask)
                iou = 1 - mask_iou_loss(q_mask, p_mask)
                fintune_top5_value.append(iou.data.item())
                finetune_eval.update_evl([class_id], q_mask.squeeze(2),
                                         p_mask.squeeze(2))

            top5_value = np.mean(top5_value)
            fintune_top5_value = np.mean(fintune_top5_value)
            top5_record.add_record(top5_key, top5_value)
            fintune_top5_record.add_record(top5_key, fintune_top5_value)
            pbar.set_postfix_str(
                'top5_key: {}, top5_value: {:.4f}, fintune_top5_value: {:.4f}'.
                format(top5_key, top5_value, fintune_top5_value))

    top5_str = top5_record.get_string()
    logger.info('{}'.format(top5_str))
    fintune_top5_str = fintune_top5_record.get_string()
    logger.info('{}'.format(fintune_top5_str))

    # test eval
    mean_f = np.mean(test_eval.f_score)
    str_mean_f = 'F: %.4f ' % (mean_f)
    mean_j = np.mean(test_eval.j_score)
    str_mean_j = 'J: %.4f ' % (mean_j)
    f_list = ['%.4f' % n for n in test_eval.f_score]
    str_f_list = ' '.join(f_list)
    j_list = ['%.4f' % n for n in test_eval.j_score]
    str_j_list = ' '.join(j_list)
    logger.info('test eval:')
    logger.info('{} {}'.format(str_mean_f, str_f_list))
    logger.info('{} {}'.format(str_mean_j, str_j_list))

    # finetune eval
    mean_f = np.mean(finetune_eval.f_score)
    str_mean_f = 'F: %.4f ' % (mean_f)
    mean_j = np.mean(finetune_eval.j_score)
    str_mean_j = 'J: %.4f ' % (mean_j)
    f_list = ['%.4f' % n for n in finetune_eval.f_score]
    str_f_list = ' '.join(f_list)
    j_list = ['%.4f' % n for n in finetune_eval.j_score]
    str_j_list = ' '.join(j_list)
    logger.info('finetune eval:')
    logger.info('{} {}'.format(str_mean_f, str_f_list))
    logger.info('{} {}'.format(str_mean_j, str_j_list))


if __name__ == '__main__':
    # 读取参数
    args = get_arguments()
    # 创建快照文件夹
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)

    args.test_dir = os.path.join(args.snapshot_dir,
                                 'test_{}'.format(args.test_num))
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)

    log_file = os.path.join(
        args.test_dir, 'test_log_{}_{}.txt'.format(args.group,
                                                   args.finetune_idx))
    print('log file: {}'.format(log_file))
    logger = Logger(log_file)
    logger.info('Running parameters:')
    logger.info(str(args))

    test(args, logger)
