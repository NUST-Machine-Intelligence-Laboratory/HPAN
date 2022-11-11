import argparse
import numpy as np
import os.path as osp
import os
import PIL.Image as Image
import cv2
from tqdm import trange

import torch
import torch.nn as nn

from libs.config.config import OPTION as opt
from libs.dataset.transform import TestTransform
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.models.HPAN.HPAN import HPAN
from libs.utils.Logger import TreeEvaluation as Evaluation


def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')

    # 文件路径设置
    # train_id = 2
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--test_id", type=int, default=1)
    parser.add_argument("--q_vid", type=int, default=1)
    parser.add_argument("--s_vid", nargs='+', type=int, default=[1])
    parser.add_argument("--snapshot_dir", type=str, default=opt.SNAPSHOTS_DIR)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=opt.DATASETS_DIR)
    parser.add_argument("--arch", type=str, default='HPAN')

    # 载入模型参数
    parser.add_argument("--snapshot_path", type=str, default=None)

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

    parser.add_argument("--output_path", type=str, default='1')

    return parser.parse_args()


def restore(model: nn.Module, snapshot_path: str):
    assert osp.exists(snapshot_path)
    checkpoint = torch.load(snapshot_path)
    if 'state_dict' in checkpoint.keys():
        weight = checkpoint['state_dict']
    else:
        weight = checkpoint
    s = model.state_dict()
    for key, val in weight.items():
        if key[:6] == 'module':
            key = key[7:]
        if key in s and s[key].shape == val.shape:
            s[key][...] = val
        elif key not in s:
            print('ignore weight from not found key {}'.format(key))
        else:
            print('ignore weight of mistached shape in key {}'.format(key))
    model.load_state_dict(s)
    print('Loaded weights from %s' % (snapshot_path))
    return weight


def visualize(cid, id, frame_num, source_path, train_id):
    target_path = source_path + '/vis'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in trange(frame_num):
        image_filename = 'image_%d_%d.png' % (id, i)
        gt_filename = 'mask_%d_%d.png' % (id, i)
        pred_filename = 'pred_%d_%d.png' % (id, i)

        image = cv2.imread(osp.join(source_path, image_filename))
        gt_mask = cv2.imread(osp.join(source_path, gt_filename))
        pred_mask = cv2.imread(osp.join(source_path, pred_filename))

        for j in [0, 2]:
            gt_mask[:, :, j] = 0 * (gt_mask[:, :, j] > 0)

        for j in [0, 1]:
            pred_mask[:, :, j] = 0 * (pred_mask[:, :, j] > 0)

        gt_combine = cv2.addWeighted(image, 0.5, gt_mask, 0.5, 0)
        pred_combine = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)

        # save
        cv2.imwrite(osp.join(target_path, 'gt_%d_%d.png' % (id, i)),
                    gt_combine)
        cv2.imwrite(
            osp.join(target_path, '%d_predict_%d_%d.png' % (train_id, id, i)),
            pred_combine)


def visualization(args):

    print('Loading model...')
    net = HPAN(backbone_type=args.backbone_type,
               with_prior_mask=args.with_prior_mask,
               with_proto_attn=args.with_proto_attn,
               proto_with_self_attn=args.proto_with_self_attn,
               proto_per_frame=args.proto_per_frame,
               with_domain_attn=args.with_domain_attn,
               domain_with_self_attn=args.domain_with_self_attn)

    train_id = args.trainid
    print('train_id: {}'.format(train_id))
    fold_id = args.group
    print('fold_id: {}'.format(fold_id))
    test_id = args.test_id
    print('test_id: {}'.format(test_id))
    test_num = args.test_num
    print('test_num: {}'.format(test_num))
    q_vid = args.q_vid
    print('q_vid: {}'.format(q_vid))
    s_vids = args.s_vid
    print('s_vids: {}'.format(s_vids))
    cid = test_id * 4 + fold_id
    print('cid: {}'.format(cid))

    restore_path = 'workdir/HPAN/id_{}_group_{}_of_4/test_{}/model_fintune_{}.pth.tar'.format(
        train_id, fold_id, test_num, cid)
    restore(net, restore_path)

    print('Loading data...')
    test_tf = TestTransform(args.input_size)
    dataset = YTVOSDataset(data_path=args.data_path,
                           test=True,
                           fold_idx=fold_id,
                           test_idx=test_id,
                           query_num=args.query_num,
                           support_num=args.support_num,
                           transforms=test_tf)
    vid_set = dataset.video_ids[0]
    print('vid_set: {}'.format(vid_set))

    q_fids = dataset.select_frame_id(cid, q_vid, test=True)
    q_frames, q_masks = dataset.get_frames_and_masks(cid, q_vid, q_fids)

    s_frames, s_masks = [], []
    for s_vid in s_vids:
        s_fids = dataset.select_frame_id(cid, s_vid, test=True)
        s_frame, s_mask = dataset.get_frames_and_masks(cid, s_vid, s_fids)
        s_frames += s_frame
        s_masks += s_mask

    q_frames_in, q_masks_in = test_tf(q_frames, q_masks)
    s_frames_in, s_masks_in = test_tf(s_frames, s_masks)

    print('q_frames_in.shape: ', q_frames_in.shape)
    print('q_masks_in.shape: ', q_masks_in.shape)
    print('s_frames_in.shape: ', s_frames_in.shape)
    print('s_masks_in.shape: ', s_masks_in.shape)

    q_frames_in = q_frames_in.unsqueeze(0)
    q_masks_in = q_masks_in.unsqueeze(0)
    s_frames_in = s_frames_in.unsqueeze(0)
    s_masks_in = s_masks_in.unsqueeze(0)

    test_eval = Evaluation([cid])
    print('Start testing...')
    net.eval()
    with torch.no_grad():
        predict_mask = net(q_frames_in, s_frames_in, s_masks_in)
    test_eval.update_evl(idx=[cid],
                         pred=predict_mask.squeeze(2),
                         query_mask=q_masks_in.squeeze(2))
    eval_str, _ = test_eval.get_eval()
    print(eval_str)

    predict_mask = predict_mask.squeeze(0)
    q_masks_in = q_masks_in.squeeze(0)
    print('q_masks_vis.shape: ', q_masks_in.shape)
    print('predict_mask.shape: ', predict_mask.shape)
    q_num = predict_mask.shape[0]

    print('video_id:', q_vid)
    video_info = dataset.vid_infos[q_vid]

    origin_image = [
        Image.open(
            os.path.join(dataset.img_dir, video_info['file_names'][frame_idx]))
        for frame_idx in range(video_info['length'])
    ]

    # save img
    output_dir = osp.join('./output', args.output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = osp.join(output_dir,
                          '{}_{}_{}_{}/'.format(train_id, cid, q_vid, q_num))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in trange(q_num):
        image_vis = origin_image[i]
        gt_mask = q_masks_in[i]
        pred_mask = predict_mask[i]

        gt_mask = gt_mask.squeeze().numpy()
        pred_mask = pred_mask.squeeze().numpy()

        gt_mask = np.uint8(gt_mask > 0.5) * 255
        pred_mask = np.uint8(pred_mask > 0.5) * 255

        save_image_path = os.path.join(output_dir,
                                       'image_%s_%d.png' % (q_vid, i))
        image_vis.thumbnail((425, 241))
        im = Image.new("RGB", (425, 241))
        w1, h1 = image_vis.size
        w2, h2 = im.size
        im.paste(image_vis, ((w2 - w1) // 2, (h2 - h1) // 2))
        im.save(save_image_path, 'png')

        save_mask_path = os.path.join(output_dir,
                                      'mask_%s_%d.png' % (q_vid, i))
        im = Image.fromarray(gt_mask)
        im.save(save_mask_path, 'png')

        save_pred_path = os.path.join(output_dir,
                                      'pred_%s_%d.png' % (q_vid, i))
        im = Image.fromarray(pred_mask)
        im.save(save_pred_path, 'png')

    visualize(cid, q_vid, q_num, output_dir, train_id)

    print('Done!')


if __name__ == '__main__':
    # 读取参数
    args = get_arguments()
    visualization(args)
