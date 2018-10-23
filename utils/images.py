from PIL import Image
import torch
import numpy as np

def imshow(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.numpy()

    arr = (tensor * 255).astype('uint8')
    img = Image.fromarray(arr)
    img.show()