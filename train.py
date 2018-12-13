import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from scene_dataset import SceneDataset, SceneSampler
import time
from nn.vae import VAEDecoder
from nn.dqn import DQN
from nn.gqn import GQN
from utils.images import imshow, imsave
import os
import json
from itertools import cycle
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_checkpoint(checkpoint):
    """
    Loads a checkpoint file for continuing the training
    Fields:
    Iteration: i
    model: file

    """
    with open(checkpoint, 'r') as f:
        dat = f.read()
    chkpt = json.loads(dat)
    print(chkpt)
    return chkpt["iteration"], chkpt["model"]

if __name__ == "__main__":
    iters = 100000
    dirname = ""

    files = os.listdir(None if not dirname else dirname)
    start_idx = 0
    model_file = None
    # if "train.chkpt" in files:
    #     start_idx, model_file = load_checkpoint(os.path.join(dirname, "train.chkpt"))

    
    start = time.perf_counter()
    # print(start)
    composed = transforms.Compose([
        transforms.Normalize([0.0855, 0.108, 0.0776], [0.268, 0.321, 0.253])
    ])

    dataset = SceneDataset(dataset='shepard_metzler_5_parts', context_size=5, root="data/general")
    val_dataset = SceneDataset(dataset='shepard_metzler_5_parts', context_size=5, root="data/general", mode='test')
    sampler = SceneSampler(dataset, start_idx=start_idx)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler,
                        num_workers=0, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=0, shuffle=False)
    # print("dataloader setup took", time.perf_counter() - start)
    iter_start = time.perf_counter()
    dataloader_iterator = cycle(iter(dataloader))
    val_iter = cycle(iter(val_dataloader))
    # print("creating iter took", time.perf_counter() -iter_start)

    model = GQN().to(device)
    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    loss_fn = nn.L1Loss()

    # lr_i = 5e-4
    # lr_f = 5e-5
    # n = 1.6e6
    # lr_func = lambda epoch: max(lr_f + (lr_i - lr_f)*(1 - epoch / n), lr_f)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func, last_epoch=start_idx-1)

    optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
    
    # , len(dataloader)
    for t in range(start_idx, iters):
        load_start = time.perf_counter()
        batch = next(dataloader_iterator)
        # print("Load took", time.perf_counter() - load_start)
        # imshow(batch['query']['context']['frames'][0][0])
        # Destruct input
        batch_context = batch['query']['context']
        batch_views = batch_context['cameras'].to(device)
        batch_frames = batch_context['frames'].to(device)        

        # Destruct Target
        batch_target_view = batch['query']['query_camera'].to(device)
        batch_target = batch['target'].to(device)
        # print(torch.max(batch_target), torch.min(batch_target))

        model.train(True)

        optimizer.zero_grad()
        forward_start = time.perf_counter()
        img = model(batch_views, batch_frames, batch_target_view)
        # print( torch.max(img[:]), torch.min(img[:]))

        # print("Forward pass took", time.perf_counter() - forward_start)
        loss_start = time.perf_counter()
        loss = loss_fn(img, batch_target)
        # print("loss calc", time.perf_counter() - loss_start)

        # p = next(model.parameters())
        # for x in p.shape[:-1]:
        #     p = p[0]
        # print(p)
        backward_start = time.perf_counter()
        loss.backward()

        # p = next(model.parameters())
        # for x in p.shape[:-1]:
        #     p = p[0]
        # print(p)

        optimizer.step()
        # print("backward took", time.perf_counter() - backward_start)
        
        # scheduler.step()
        # print("iter took", time.perf_counter() - load_start)

        if t % 10 == 0:
            model.eval()
            batch = next(val_iter)
 
            # Destruct input
            batch_context = batch['query']['context']
            batch_views = batch_context['cameras'].to(device)
            batch_frames = batch_context['frames'].to(device)        

            # Destruct Target
            batch_target_view = batch['query']['query_camera'].to(device)
            batch_target = batch['target'].to(device)

            img = model(batch_views, batch_frames, batch_target_view)
            loss = loss_fn(img, batch_target)

            print('[%5d]'%t, 'loss = %f'%loss)
        
            # imshow(img[0].detach())
            if t % 100 == 0:
                imsave(batch, img.detach(), "predictions/iteration_{:05d}.png".format(t))
                path = os.path.join(dirname, 'dqn-larger.th')
                torch.save(model.state_dict(), path) # Do NOT modify this line
                json_dat = json.dumps({"iteration": t, "model": path})
                with open( os.path.join(dirname, "train.chkpt"), 'w') as f:
                    f.write(json_dat)






        

    end = time.perf_counter()
    print("end", end)
    dur = end - start
    print("dur", dur)


