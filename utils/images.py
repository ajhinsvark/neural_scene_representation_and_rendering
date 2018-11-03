from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def imshow(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.numpy()

    arr = (tensor * 255).astype('uint8')
    img = Image.fromarray(arr)
    img.show()

def imsave(batch, prediction, fname):
    rand_idx = random.randrange(len(batch['target']))
    # Destruct input
    batch_context = batch['query']['context']
    batch_views = batch_context['cameras'][rand_idx]
    batch_frames = batch_context['frames'][rand_idx]
    input_size = len(batch_frames)

    # Destruct Target
    batch_target_view = batch['query']['query_camera'][rand_idx]
    batch_target = batch['target'][rand_idx]

    fig_height = max((input_size + 1) // 2, 2)
    fig, axs = plt.subplots(fig_height, 3, squeeze=False)

    # Plot the context images
    for i in range(input_size):
        im = batch_frames[i]
        ax = axs[i // 2, i % 2]
        ax.imshow(im, interpolation='bilinear')
        ax.set_title("Context Image {}".format(i))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    if input_size % 2 == 1:
        fig.delaxes(axs[input_size // 2, 1])
    
    for i in range(2, fig_height):
        fig.delaxes(axs[i,2])
    
    axs[0,2].imshow(batch_target, interpolation='bilinear')
    axs[0,2].set_title("Target")
    axs[0,2].get_xaxis().set_visible(False)
    axs[0,2].get_yaxis().set_visible(False)
    axs[0,2].set_yticklabels([])
    axs[0,2].set_xticklabels([])

    axs[1,2].set_title("Prediction")
    axs[1,2].imshow(prediction[rand_idx], interpolation='bilinear')
    axs[0,2].get_xaxis().set_visible(False)
    axs[0,2].get_yaxis().set_visible(False)
    axs[1,2].set_yticklabels([])
    axs[1,2].set_xticklabels([])

    plt.savefig("test_fig.png")
        