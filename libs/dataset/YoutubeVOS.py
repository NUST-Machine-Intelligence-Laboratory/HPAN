from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image


class YTVOSDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 train=False,
                 valid=False,
                 test=False,
                 fold_idx=1,
                 test_idx=None,
                 query_num=5,
                 support_num=5,
                 sample_per_class=100,
                 transforms=None):
        assert train ^ valid ^ test
        assert fold_idx in [1, 2, 3, 4]
        assert (not test) ^ (test_idx in range(10))
        assert support_num > 0 and query_num > 0
        self.train = train
        self.valid = valid
        self.test = test
        self.fold_idx = fold_idx
        self.support_num = support_num
        self.query_num = query_num
        self.sample_per_class = sample_per_class
        self.transforms = transforms

        if data_path is None:
            data_path = os.path.join('/your_path_to_data/data')
        data_dir = os.path.join(data_path, 'Youtube-VOS')
        self.img_dir = os.path.join(data_dir, 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'train', 'train.json')

        self.load_annotations()

        print('data fold index: ', fold_idx)
        train_list = [n + 1 for n in range(40) if n % 4 != (fold_idx - 1)]
        valid_list = [n + 1 for n in range(40) if n % 4 == (fold_idx - 1)]

        if train:
            self.class_list = train_list
        elif valid:
            self.class_list = valid_list
        elif test:
            self.class_list = [valid_list[test_idx]]

        self.video_ids = []
        for class_id in self.class_list:
            tmp_list = self.ytvos.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)

        if test:
            self.length = len(self.video_ids[0]) - support_num
        else:
            self.length = len(self.class_list) * sample_per_class

        self.max_area_ids = {}
        for i in range(len(self.class_list)):
            class_id = self.class_list[i]
            for vid_id in self.video_ids[i]:
                anno_ids = self.ytvos.getAnnIds(vidIds=vid_id, catIds=class_id)
                anno_infos = self.ytvos.loadAnns(anno_ids)
                anno_areas = [anno['areas'] for anno in anno_infos]
                anno_areas_sum = []
                for frame_id in range(self.vid_infos[vid_id]['length']):
                    sum = 0
                    for anno_area in anno_areas:
                        if anno_area[frame_id] is not None:
                            sum += anno_area[frame_id]
                    anno_areas_sum.append(sum)
                max_area_id = anno_areas_sum.index(max(anno_areas_sum))
                self.max_area_ids[vid_id] = max_area_id

    def load_annotations(self):
        self.ytvos = YTVOS(self.ann_file)
        self.vid_ids = self.ytvos.getVidIds()  # list[2238] begin : 1
        self.vid_infos = self.ytvos.vids  # vids
        for vid, vid_info in self.vid_infos.items():  # for each vid
            vid_name = vid_info['file_names'][0].split('/')[0]  # '0043f083b5'
            vid_info['dir'] = vid_name
            frame_len = vid_info['length']  # int
            frame_object, frame_class = [], []
            for i in range(frame_len):
                frame_object.append([])
            for i in range(frame_len):
                frame_class.append([])
            category_set = set()
            annos = self.ytvos.vidToAnns[vid]  # list[]
            for anno in annos:  # instance_level anns
                assert len(anno['segmentations']) == frame_len, (
                    vid_name, len(anno['segmentations']), vid_info['length'])
                for frame_idx in range(frame_len):
                    anno_segmentation = anno['segmentations'][frame_idx]
                    if anno_segmentation is not None:
                        frame_object[frame_idx].append(
                            anno['id'])  # add instance to vid_frame
                        frame_class[frame_idx].append(
                            anno['category_id']
                        )  # add instance class to vid_frame
                        category_set = category_set.union(
                            {anno['category_id']})
            vid_info['objects'] = frame_object
            vid_info['classes'] = frame_class
            class_frame_id = dict()
            for class_id in category_set:  # frames index for each class
                class_frame_id[class_id] = [
                    i for i in range(frame_len) if class_id in frame_class[i]
                ]
            vid_info['class_frames'] = class_frame_id

    def get_frame_by_vid(self, video_id, choice_frame):
        vid_info = self.vid_infos[video_id]
        frames = [
            np.array(
                Image.open(
                    os.path.join(self.img_dir,
                                 vid_info['file_names'][frame_idx])))
            for frame_idx in choice_frame
        ]
        return frames

    def get_mask_by_cid_and_vid(self, class_id, video_id, choice_frame):
        vid_info = self.vid_infos[video_id]
        masks = []
        for frame_id in choice_frame:
            object_ids = vid_info['objects'][frame_id]
            mask = None
            for object_id in object_ids:
                ann = self.ytvos.loadAnns(object_id)[0]
                if ann['category_id'] not in self.class_list:
                    continue
                track_id = 1
                if ann['category_id'] != class_id:
                    track_id = 0
                temp_mask = self.ytvos.annToMask(ann, frame_id)
                if mask is None:
                    mask = temp_mask * track_id
                else:
                    mask += temp_mask * track_id
            assert mask is not None
            mask[mask > 0] = 1
            masks.append(mask)
        return masks

    def get_frames_and_masks(self, class_id, video_id, choice_frame):
        frames = self.get_frame_by_vid(video_id, choice_frame)
        masks = self.get_mask_by_cid_and_vid(class_id, video_id, choice_frame)
        return frames, masks

    def select_frame_id(self,
                        class_id,
                        video_id,
                        train=False,
                        test=False,
                        support=False):
        assert train ^ test ^ support
        vid_info = self.vid_infos[video_id]
        frame_list = vid_info['class_frames'][class_id]
        frame_len = len(frame_list)
        if test:
            frame_ids = frame_list
        elif support:
            frame_ids = random.sample(frame_list, 1)
            # frame_ids = [self.max_area_ids[video_id]
            #              ] if self.test else frame_ids
        else:
            frame_num = self.query_num
            if frame_len > frame_num:
                start_frame = random.randint(0, frame_len - frame_num)
                frame_ids = frame_list[start_frame:start_frame + frame_num]
            elif frame_len == frame_num:
                frame_ids = frame_list
            else:
                frame_ids = frame_list + [
                    frame_list[-1] for _ in range(frame_num - frame_len)
                ]
        return frame_ids

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class
        class_id = self.class_list[list_id]
        vid_set = self.video_ids[list_id]

        choice_vids = random.sample(vid_set, 1 + self.support_num)
        q_video_id = choice_vids[0]
        s_video_ids = choice_vids[1:]

        q_frame_ids = self.select_frame_id(class_id, q_video_id, train=True)
        q_frames, q_masks = self.get_frames_and_masks(class_id, q_video_id,
                                                      q_frame_ids)

        s_frame_ids, s_frames, s_masks = [], [], []
        for support_vid in s_video_ids:
            s_frame_id = self.select_frame_id(class_id,
                                              support_vid,
                                              support=True)
            one_frame, one_mask = self.get_frames_and_masks(
                class_id, support_vid, s_frame_id)
            s_frame_ids += s_frame_id
            s_frames += one_frame
            s_masks += one_mask

        if self.transforms is not None:
            q_frames, q_masks = self.transforms(q_frames, q_masks)
            s_frames, s_masks = self.transforms(s_frames, s_masks)

        q_info = {}
        q_info['class_id'] = class_id
        q_info['video_id'] = q_video_id
        q_info['frame_ids'] = q_frame_ids
        s_info = {}
        s_info['class_id'] = class_id
        s_info['video_ids'] = s_video_ids
        s_info['frame_ids'] = s_frame_ids
        return q_frames, q_masks, s_frames, s_masks, q_info, s_info

    def __gettestitem__(self, idx):
        class_id = self.class_list[0]
        begin_new = (idx == 0)
        if begin_new:
            vid_set = self.video_ids[0]
            self.test_s_vids = random.sample(vid_set, self.support_num)
            self.test_q_vids = [
                vid for vid in vid_set if vid not in self.test_s_vids
            ]
        q_video_id = self.test_q_vids[idx]
        s_video_ids = self.test_s_vids

        q_frame_ids = self.select_frame_id(class_id, q_video_id, test=True)
        q_frames, q_masks = self.get_frames_and_masks(class_id, q_video_id,
                                                      q_frame_ids)

        s_frame_ids, s_frames, s_masks = [], [], []
        if begin_new:
            for support_vid in s_video_ids:
                s_frame_id = self.select_frame_id(class_id,
                                                  support_vid,
                                                  support=True)
                one_frame, one_mask = self.get_frames_and_masks(
                    class_id, support_vid, s_frame_id)
                s_frame_ids += s_frame_id
                s_frames += one_frame
                s_masks += one_mask

        if self.transforms is not None:
            q_frames, q_masks = self.transforms(q_frames, q_masks)
            if begin_new:
                s_frames, s_masks = self.transforms(s_frames, s_masks)

        q_info = {}
        q_info['class_id'] = class_id
        q_info['video_id'] = q_video_id
        q_info['frame_ids'] = q_frame_ids
        s_info = {}
        s_info['class_id'] = class_id
        s_info['video_ids'] = s_video_ids
        s_info['frame_ids'] = s_frame_ids
        return q_frames, q_masks, s_frames, s_masks, q_info, s_info, begin_new

    def __getitem__(self, idx):
        if self.test:
            return self.__gettestitem__(idx)
        else:
            return self.__gettrainitem__(idx)

    def __len__(self):
        return self.length

    def get_class_list(self):
        return self.class_list
