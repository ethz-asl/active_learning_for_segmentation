import torch.utils.data as torchData
import numpy as np

class CombinedDataset(torchData.Dataset):
  """ Dataset that combines multiple torch datasets (e.g. 1 replay set + 1 Training set) """
  def __init__(self, datasets, transform=None):
    super().__init__()
    self.datasets = datasets
    self.transform = transform

    lengths = [len(d) for d in datasets]

    # Calculate length of this dataset as sum of lengths of all dasets
    self._length = 0
    for l in lengths:
      self._length += l

    # Randomly assign each requested idx to a dataset, making sure to not have diversity of images in batches
    idxs = np.asarray([idx for _ in range(l) for idx, l in enumerate(lengths)])
    permuted_idxs = np.random.permutation(idxs)
    self.idx_to_ds = permuted_idxs

  def __getitem__(self, index):
    # Dataset internal index (e.g. if first two images are from ds1 and next two are from ds2, index 2 would be mapped
    # to internal idx 0 and dataset index 1
    ds_internal_idx = np.sum(self.idx_to_ds[0: index] == self.idx_to_ds[index])
    data = self.datasets[self.idx_to_ds[index]][ds_internal_idx]

    # Apply transformations
    if self.transform is not None:
      data = self.transform(data)

    return data

  def __len__(self):
    return self._length
