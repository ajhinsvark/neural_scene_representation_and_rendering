# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal data reader for GQN TFRecord datasets."""

import collections
import os
import tensorflow as tf
import numpy as np
from PIL import Image

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5


def _get_dataset_files(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dateset_info.train_size
  else:
    num_files = dateset_info.test_size

  length = len(str(num_files))
  template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
  return [os.path.join(base, template.format(i + 1, num_files))
          for i in range(num_files)]


def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)

def _get_randomized_indices(dataset_info, example_size):
    """Generates randomized indices into a sequence of a specific length."""
    indices = tf.range(0, dataset_info.sequence_size)
    indices = tf.random_shuffle(indices)
    indices = tf.slice(indices, begin=[0], size=[example_size])
    return indices

def _preprocess_frames(example, indices, dataset_info, example_size, custom_frame_size):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.gather(frames, indices, axis=0)
    frames = tf.map_fn(
        _convert_frame_data, tf.reshape(frames, [-1]),
        dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
        [dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    # tf.Print(tf.shape(frames), [tf.shape(frames)], "Shape: ")
    frames = tf.reshape(
        frames, (-1, example_size) + dataset_image_dimensions)
    # tf.Print(tf.shape(frames), [tf.shape(frames)], "Shape: ")

    if (custom_frame_size and
        custom_frame_size != dataset_info.frame_size):
      frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
      new_frame_dimensions = (custom_frame_size,) * 2 + (_NUM_CHANNELS,)
      frames = tf.image.resize_bilinear(
          frames, new_frame_dimensions[:2], align_corners=True)
      frames = tf.reshape(
          frames, (-1, example_size) + new_frame_dimensions)
    return frames

def _preprocess_cameras(example, indices, dataset_info):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
        [pos, yaw, pitch], axis=2)
    return cameras

def _parse_function(example_proto, dataset_info, example_size, custom_frame_size):
    feature_map = {
        'frames': tf.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.FixedLenFeature(
            shape=[dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
            dtype=tf.float32)
    }
    example = tf.parse_single_example(example_proto, feature_map)
    indices = _get_randomized_indices(dataset_info, example_size)
    frames = _preprocess_frames(example, indices, dataset_info, example_size, custom_frame_size)
    cameras = _preprocess_cameras(example, indices, dataset_info)
    return frames, cameras

def make_dataset(dataset, root, context_size=5, mode='train', custom_frame_size=None, load_all=False):
    dataset_info = _DATASETS[dataset]
    file_names = _get_dataset_files(dataset_info, mode, root)
    dataset = tf.data.TFRecordDataset(file_names)
    if load_all:
        context_size = dataset_info.sequence_size - 1
    def parse_func(example_proto):
        return _parse_function(example_proto, dataset_info=dataset_info, example_size=context_size + 1, custom_frame_size=custom_frame_size)
    dataset = dataset.map(parse_func)
    dataset = dataset.repeat(1)
    return dataset

class DatasetWriter:

    def __init__(self, dataset, mode, root):
        """
        Writes images to files, and camera info csv
        """
        self.dataset_info = _DATASETS[dataset]
        self.mode = mode
        self.root = root
        self.counter = 0
        # csv_header = "scene,view,x,y,z,yaw,pitch"
        path = os.path.join(self.root, self.dataset_info.basepath, self.mode)
        os.makedirs(path, exist_ok=True)
        self.meta_file = os.path.join(path, "info.meta")
        # self.csv = os.path.join(path, "info.csv")
        # with open(self.csv, 'w+') as f:
        #     f.write(csv_header)
            

    def save_multiple(self, records):
        img_dir = os.path.join(self.root, self.dataset_info.basepath, self.mode)
        frames = []
        cameras = []
        for rec_frames, rec_cameras in records:
            rec_frames = np.squeeze(rec_frames)
            rec_cameras = np.squeeze(rec_cameras)
            frames.append(rec_frames)
            cameras.append(rec_cameras)
        frames = np.array(frames)
        cameras = np.array(cameras)
        # np.savez(os.path.join(img_dir, "record-{}.npy".format(self.counter + 1)), frames=frames, cameras=cameras)
        np.savez_compressed(os.path.join(img_dir, "record-{}.npz".format(self.counter + 1)), frames=frames, cameras=cameras)
        self.counter += 1
        if self.mode == 'train':
            num_files = 2e6 * 9 / 10
        else:
            num_files = 2e6 * 1 / 10

        if self.counter % 1000 == 0:
            print(  "{}% complete".format(self.counter * 100 / num_files))




    def save_record(self, record):
        # image1, context1, image2, context2, ..., imageN, contextN
        # context1 = x1, y1, z1, yaw1, pitch1
        frames, cameras = record
        img_dir = os.path.join(self.root, self.dataset_info.basepath, self.mode)
        try:
            frames = np.squeeze(frames)
            cameras = np.squeeze(cameras)
        except ValueError:
            pass
        if self.mode == 'train':
            num_files = 2e6 * 9 / 10
        else:
            num_files = 2e6 * 1 / 10

        rows = []
        if self.counter % 1000 == 0:
            print("{}% complete".format(self.counter * 100 / num_files))
        
        self.counter += 1

