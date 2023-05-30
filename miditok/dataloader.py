import numpy as np
import os
import torch

class DataLoader:

  def __init__(self, path, block_size=1024):
    self.data = np.memmap(path, dtype=np.uint16, mode='r')
    self.block_size = block_size

  def __len__(self):
    # Return the number of items in your dataset.
    return len(self.data)

  def __getitem__(self, value):
    data = self.data
    ix = torch.randint(len(data) - self.block_size, (1,))
    i = ix[0]
    x = torch.from_numpy((data[i:i+self.block_size]).astype(np.int64))
    y = x

    return {'input_ids': x, 'labels': y}
