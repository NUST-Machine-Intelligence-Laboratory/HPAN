
import sys
import time

import numpy as np
import torch
from libs.utils.davis_JF import db_eval_boundary, db_eval_iou


class Logger(object):
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        self.log_file = open(log_file_name, 'w')
        self.log_file.write('%s\n' % self._get_time_str())

    def __del__(self):
        self.log_file.close()

    def _get_time_str(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def info(self, msg):
        self.log_file.write('%s\n' % msg)
        self.log_file.flush()
        print(msg)


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count


class Loss_record():
    '''save the loss: total(tensor), part1 and part2 can be 0'''
    def __init__(self, update_speed=0.01):
        self.loss_dict = {}
        self.update_speed = update_speed

    def updateloss(self, new_loss_dict):

        for key in new_loss_dict:
            loss = new_loss_dict[key].data.item()
            if key not in self.loss_dict:
                self.loss_dict[key] = loss
            else:
                self.loss_dict[key] = (1 - self.update_speed) * self.loss_dict[
                    key] + self.update_speed * loss

    def getloss(self):
        ''' get every step loss and reset '''
        out_str = ''
        for key in self.loss_dict:
            out_str += '%s: %.4f ' % (key, self.loss_dict[key])
        return out_str


def measure(y_in, pred_in):
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


class TreeEvaluation():
    '''eval training output'''
    def __init__(self, class_list=None):
        assert class_list is not None
        self.class_indexes = class_list
        self.num_classes = len(class_list)
        self.setup()

    def setup(self):
        self.tp_list = [0] * self.num_classes
        self.f_list = [0] * self.num_classes
        self.j_list = [0] * self.num_classes
        self.n_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes

        self.f_score = [0] * self.num_classes
        self.j_score = [0] * self.num_classes

    def update_evl(self, idx, query_mask, pred):
        # B N H W
        batch = len(idx)
        for i in range(batch):
            if not isinstance(idx[i], int):
                id = idx[i].item()
            else:
                id = idx[i]
            id = self.class_indexes.index(id)
            tp, total = self.test_in_train(query_mask[i], pred[i])
            for j in range(query_mask[i].shape[0]):
                thresh = .5
                y = query_mask[i][j].cpu().numpy() > thresh
                predict = pred[i][j].data.cpu().numpy() > thresh
                self.f_list[id] += db_eval_boundary(predict, y)
                self.j_list[id] += db_eval_iou(y, predict)
                self.n_list[id] += 1

            self.tp_list[id] += tp
            self.total_list[id] += total
        self.iou_list = [
            self.tp_list[ic] / float(max(self.total_list[ic], 1))
            for ic in range(self.num_classes)
        ]
        self.f_score = [
            self.f_list[ic] / float(max(self.n_list[ic], 1))
            for ic in range(self.num_classes)
        ]
        self.j_score = [
            self.j_list[ic] / float(max(self.n_list[ic], 1))
            for ic in range(self.num_classes)
        ]

    def test_in_train(self, query_label, pred):
        # test N*H*F
        pred = pred.data.cpu().numpy()
        query_label = query_label.cpu().numpy()

        tp, tn, fp, fn = measure(query_label, pred)
        total = tp + fp + fn
        return tp, total

    def get_eval(self):
        mean_iou = np.mean(self.iou_list)
        out_str = 'iou: %.4f' % mean_iou
        mean_f = np.mean(self.f_score)
        out_str += ' f: %.4f' % mean_f
        mean_j = np.mean(self.j_score)
        out_str += ' j: %.4f' % mean_j
        self.setup()
        return out_str, [mean_iou, mean_f, mean_j]


class TimeRecord():
    def __init__(self, max_epoch, max_iter):
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.start_time = time.time()

    def get_time(self, epoch, iter):
        now_time = time.time()
        total_time = now_time - self.start_time
        total_iter = epoch * self.max_iter + iter
        remain_iter = self.max_epoch * self.max_iter - total_iter
        remain_time = remain_iter * total_time / total_iter
        # time_str "xx:xx:xx"
        total_time_str = '{:02d}:{:02d}:{:02d}'.format(
            int(total_time // 3600), int(total_time % 3600 // 60),
            int(total_time % 60))
        remain_time_str = '{:02d}:{:02d}:{:02d}'.format(
            int(remain_time // 3600), int(remain_time % 3600 // 60),
            int(remain_time % 60))
        return total_time_str, remain_time_str


class LogTime():
    def __init__(self):
        self.reset()

    def t1(self):
        self.logt1 = time.time()

    def t2(self):
        self.logt2 = time.time()
        self.alltime += (self.logt2 - self.logt1)

    def reset(self):
        self.logt1 = None
        self.logt2 = None
        self.alltime = 0

    def getalltime(self):
        out = self.alltime
        self.reset()
        return out


class TopFiveRecord:
    def __init__(self):
        self.key_list = []
        self.value_list = []

    def add_record(self, key: str, value):
        self.key_list.append(key)
        self.value_list.append(value)
        self.sort_record()

    def sort_record(self):
        self.key_list, self.value_list = (list(t) for t in zip(
            *sorted(zip(self.key_list, self.value_list),
                    key=lambda x: x[1],
                    reverse=True)))

    def get_top_five(self):
        return [self.key_list[:5], self.value_list[:5]]

    def get_down_five(self):
        return [self.key_list[-5:], self.value_list[-5:]]

    def get_string(self):
        result_str = ""
        top_5 = self.get_top_five()
        down_5 = self.get_down_five()
        result_str += "Top 5: \n"
        for i in range(5):
            result_str += "{} {}\n".format(top_5[0][i], top_5[1][i])
        result_str += "Down 5: \n"
        for i in range(5):
            result_str += "{} {}\n".format(down_5[0][i], down_5[1][i])
        return result_str
