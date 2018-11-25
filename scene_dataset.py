import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
from skimage import io
import numpy as np
import numpy.random
import os
import collections
import time
import random

from utils.dataset_reader import _DATASETS

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])

def process_camera(camera):
    pos = camera[:, :3]
    yaw = camera[:, 3:4]
    pitch = camera[:, 4:5] 
    return np.concatenate([pos, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)], axis=1)


class SceneSampler(Sampler):

    def __init__(self, data_source, start_idx=0):
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(range(self.start_idx, len(self.data_source)))


class SceneDataset(Dataset):

    def __init__(self, dataset, context_size, root, mode='train', transform=None):
        self.dataset_info = _DATASETS[dataset]
        self.path = os.path.join(root, self.dataset_info.basepath, mode)
        self.context_size = context_size
        with open(os.path.join(self.path, 'info.meta'), 'r') as f:
            self.len = int(f.read())
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        rem = i % 1000
        name = "record-{}.npz".format(i // 1000 + 1)    
        file_path = os.path.join(self.path, name)
        load_start = time.perf_counter()
        f = np.load(file_path, mmap_mode=None)
        # print("load file", name, "took", time.perf_counter() - load_start)
        frames = f['frames']
        cameras = f['cameras']

        scene_frames = frames[rem]
        scene_cameras = cameras[rem]
        permute_start = time.perf_counter()
        sample_inds = np.random.permutation(len(scene_frames))[:self.context_size + 1]
        # print("permute", time.perf_counter() - permute_start)
        sample_frames = scene_frames[sample_inds]
        if self.transform:
            sample_frames = [self.transform(img) for img in sample_frames]
        process_start = time.perf_counter()
        sample_cameras = process_camera(scene_cameras[sample_inds])
        # print("processing", time.perf_counter() - process_start)

        # Organize query and context
        context_frames = sample_frames[:-1]
        target = sample_frames[-1]
        context_cameras = sample_cameras[:-1]
        query_camera = sample_cameras[-1]

        context = dict(frames=context_frames, cameras=context_cameras)
        query = dict(context=context, query_camera=query_camera)
        return dict(query=query, target=target)

        # template = "scene-%d-view-{}.jpg" % (i)
        # scene_df = self.views_df[self.views_df['scene'] == i]
        # print(scene_df.info())
        # views = []
        # cameras = []
        # shuffled_df = scene_df.sample(frac=1)
        # for _, _, view, *camera in shuffled_df[:self.context_size + 1].itertuples():
        #     views.append(view)
        #     cameras.append(process_camera(camera))
        # cameras = np.array(cameras)
        # imgs = []
        # for view in views:
        #     img_file = template.format(view)
        #     img = io.imread(os.path.join(self.path, img_file))
        #     img = (img / 255).astype('float32')
        #     imgs.append(img)

        # # Organize query and context
        # frames = np.stack(imgs, axis=0)
        # context_frames = frames[:-1]
        # target = frames[-1]
        # context_cameras = cameras[:-1]
        # query_camera = cameras[-1]

        # context = dict(frames=context_frames, cameras=context_cameras)
        # query = dict(context=context, query_camera=query_camera)
        # return dict(query=query, target=target)
        