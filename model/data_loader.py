# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import math
import json
from scipy.interpolate import interp1d
# from model.f1_score_test import interpolate_pred
from f1_score_test import interpolate_pred, interpolate_features


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index, config):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        self.interpolation_kind = 'nearest' 
        self.interpolate_frame_features = config.interpolate_frame_features
        self.n_shots = config.n_shots
        self.n_gt = config.n_gt
        self.num_augmentation_windows = config.num_augmentation_windows

        ### filename is h5
        # /{key}
        # TODO account for step_size != stack_size when generating video features
        #    /features (n_features, 1024)
        #    /heat-markers (100)
        
        # heat_markers need to be interpolated to n_features


        self.filename = './data/datasets/'+ self.name + '/out.h5'
        self.splits_filename = ['./data/datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores, self.list_video_names = [], [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = np.array(hdf[video_name + '/features'])
            heat_markers = np.array(hdf[video_name + '/heat-markers'])

            heat_markers_orig_len = heat_markers.shape[0]
            if self.n_shots and self.n_shots != heat_markers_orig_len: # average heat markers over n_shots bins
                heat_markers = interpolate_pred(heat_markers, self.n_shots)
            if self.n_gt and self.n_gt != heat_markers.shape[0]:
                step = heat_markers_orig_len / heat_markers.shape[0]
                shots_heat_markers_space = np.linspace(0.5*step, heat_markers_orig_len-0.5*step, heat_markers.shape[0])
                gt_step = heat_markers_orig_len / self.n_gt
                gt_space = np.linspace(0.5*gt_step, heat_markers_orig_len-0.5*gt_step, self.n_gt)
                heat_markers = interp1d(shots_heat_markers_space, heat_markers, kind='nearest', assume_sorted=True, fill_value='extrapolate')(gt_space)
            
            heat_markers_space = np.linspace(0.5, heat_markers.shape[0]-0.5, heat_markers.shape[0])
            step = heat_markers.shape[0] / frame_features.shape[0]
            frame_features_heat_markers_space = np.linspace(0.5*step, heat_markers.shape[0]-0.5*step, frame_features.shape[0])
            if self.interpolate_frame_features:
                # xp = frame_features_heat_markers_space
                # x = heat_markers_space
                # frame_features = interp1d(xp, frame_features, axis=0, assume_sorted=True, fill_value='extrapolate')(x)
                frame_features = interpolate_features(frame_features, n=heat_markers.shape[0])
                gtscore = heat_markers
            else:
                xp = heat_markers_space
                x = frame_features_heat_markers_space
                gtscore = interp1d(xp, heat_markers, kind=self.interpolation_kind, assume_sorted=True, fill_value='extrapolate')(x)

            if isinstance(self.num_augmentation_windows, int) and self.num_augmentation_windows > 0:
                # print(f"Using {self.num_augmentation_windows} augmentation windows")
                relative_window_sizes = [0.5]
                for relative_window_size in relative_window_sizes:
                    n_windows = self.num_augmentation_windows
                    n_features = frame_features.shape[0]
                    window_size = math.floor(n_features * relative_window_size)
                    window_starts = np.arange(0, n_windows) * int((n_features-window_size)//n_windows)
                    for window_start in window_starts:
                        window_end = window_start + window_size
                        self.list_frame_features.append(torch.Tensor(frame_features[window_start:window_end]))
                        self.list_gtscores.append(torch.Tensor(gtscore[window_start:window_end]))
                        self.list_video_names.append(video_name)

            self.list_frame_features.append(torch.Tensor(frame_features))
            self.list_gtscores.append(torch.Tensor(gtscore))
            self.list_video_names.append(video_name)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        # self.len = len(self.split[self.mode+'_keys'])
        return self.list_frame_features.__len__()

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        # video_name = self.split[self.mode + '_keys'][index]
        video_name = self.list_video_names[index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]

        # if self.mode == 'test':
        #     return frame_features, gtscore, video_name
        # else:
        return frame_features, gtscore, video_name


def get_loader(mode, video_type, split_index, config):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode.lower(), video_type, split_index, config)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode.lower(), video_type, split_index, config)


if __name__ == '__main__':
    pass
