import random
from typing import Generator, Iterable, List

from embodied_active_learning_core.online_learning.sample import TrainSample


def batch(iterable, n=1) -> Iterable[TrainSample]:
  """ Helper function that creates baches from any iterable """
  if n <= 0:
    yield iterable[:]
  else:
    l = len(iterable)
    for ndx in range(0, l, n):
      yield iterable[ndx:min(ndx + n, l)]


def importance_sampling_by_uncertainty(all_samples: List[TrainSample], N=250):
  """  Resample N elements from all_samples. Samples with higher uncertainty have a higher chance to be drown
  uses the 'uncertainty' property of the items in all_samples
  """
  ret_list = []
  sum_probabilities = 0
  for e in all_samples:
    sum_probabilities += e.uncertainty

  skip_idx = []
  for _ in range(N):
    r = sum_probabilities * random.random()
    sum_until_now = 0
    for n, entry in enumerate(all_samples):
      if n in skip_idx:
        # already sampled this element
        continue

      if (sum_until_now < r and ((sum_until_now + entry.uncertainty) >= r)) or sum_probabilities == 0:
        # sample this element.
        ret_list.append(entry)
        # remove this entry
        sum_probabilities -= entry.uncertainty
        skip_idx.append(n)
        break

      sum_until_now += entry.uncertainty

  return ret_list
