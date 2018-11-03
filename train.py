import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from scene_dataset import SceneDataset
import time
from nn.vae import VAEDecoder
from nn.dqn import DQN
from nn.gqn import GQN
from utils.images import imshow
import os
from itertools import cycle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
printf("Running on", device)
if __name__ == "__main__":

    dirname = ""

    iters = 10000
    start = time.perf_counter()
    # print(start)
    dataset = SceneDataset(dataset='shepard_metzler_5_parts', context_size=5, root="data/general")
    dataloader = DataLoader(dataset, batch_size=36,
                        num_workers=8, shuffle=True)
    # print("dataloader setup took", time.perf_counter() - start)
    iter_start = time.perf_counter()
    dataloader_iterator = cycle(iter(dataloader))
    # print("creating iter took", time.perf_counter() -iter_start)

    model = GQN().to(device)

    loss_fn = nn.L1Loss()

    lr_i = 5e-4
    lr_f = 5e-5
    n = 1.6e6
    lr_func = lambda epoch: max(lr_f + (lr_i - lr_f)*(1 - epoch / n), lr_f)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func, last_epoch=-1)
    

    for t in range(iters):
        load_start = time.perf_counter()
        batch = next(dataloader_iterator).to(device)
        # print("Load took", time.perf_counter() - load_start)
        # imshow(batch['query']['context']['frames'][0][0])
        # Destruct input
        batch_context = batch['query']['context']
        batch_views = batch_context['cameras']
        batch_frames = batch_context['frames']

        # Destruct Target
        batch_target_view = batch['query']['query_camera']
        batch_target = batch['target']

        model.train(True)

        optimizer.zero_grad()
        forward_start = time.perf_counter()
        img = model(batch_views, batch_frames, batch_target_view)
        # print("Forward pass took", time.perf_counter() - forward_start)
        loss_start = time.perf_counter()
        loss = loss_fn(img, batch_target)
        # print("loss calc", time.perf_counter() - loss_start)

        backward_start = time.perf_counter()
        loss.backward()
        # print("backward took", time.perf_counter() - backward_start)
        
        scheduler.step()
        # print("iter took", time.perf_counter() - load_start)

        if t % 10 == 0:
            print('[%5d]'%t, 'loss = %f'%loss)
            if t % 100 == 0:
                imshow(img[0].detach())
                torch.save(model.state_dict(), os.path.join(dirname, 'convnet.th')) # Do NOT modify this line






        

    end = time.perf_counter()
    print(end)
    dur = end - start
    print(dur)