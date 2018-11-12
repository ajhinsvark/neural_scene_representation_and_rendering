import torch
from scene_dataset import SceneDataset
import numpy as np
import time

dataset = SceneDataset(dataset='shepard_metzler_5_parts', context_size=5, root="data/general")

p = 0.00001
sample_size = int(p * len(dataset))
print("Samples:", sample_size)

inds = np.random.choice(len(dataset), sample_size)
scene = dataset[inds[0]]
imgs = np.array(scene['query']['context']['frames'] + scene['target'])
for i in inds[1:]:
    # start = time.perf_counter()
    scene = dataset[i]
    # print("Load took", time.perf_counter() - start)
    imgs = np.concatenate([imgs, np.array(scene['query']['context']['frames'] + scene['target'])], axis=0)
    
imgs = np.transpose(imgs, [3, 0, 1, 2])
imgs = np.reshape(imgs, (3, -1))
stds =  np.std(imgs, axis=1)
means = np.mean(imgs, axis=1)
print(means, stds)
