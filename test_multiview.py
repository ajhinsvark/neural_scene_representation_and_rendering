import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from scene_dataset import SceneDataset
from utils.images import imsave
import time
from nn.gqn import GQN
import os
from itertools import cycle
import numpy as np
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    dirname = ""

    composed = transforms.Compose([
        transforms.Normalize([0.0855, 0.108, 0.0776], [0.268, 0.321, 0.253])
    ])

    dataset = SceneDataset(dataset='shepard_metzler_5_parts', context_size=5, root="data/general", mode='test')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    
    dataloader_iterator = cycle(iter(dataloader))

    model = GQN().to(device)
    if args.model:
        print("Loading model file")
        model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    
    model.eval()
    for t in range(10):
        batch = next(dataloader_iterator)

        # Destruct input
        batch_context = batch['query']['context']
        batch_views = batch_context['cameras'].to(device)
        batch_frames = batch_context['frames'].to(device)        

        # Destruct Target
        batch_target_view = batch['query']['query_camera'].to(device)
        batch_target = batch['target'].to(device)

        img = model(batch_views, batch_frames, batch_target_view)

        imsave(batch, img.detach(), "tests/{:05d}.png".format(t))


