"""Nyu_depth_v2_labeled dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# TODO(Nyu_depth_v2_labeled): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The NYU-Depth V2 labeled data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
It contains 1449 densely labeled pairs of aligned RGB and depth images.
"""

# TODO(Nyu_depth_v2_labeled): BibTeX citation
_CITATION = """
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
"""

_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'


class NyuDepthFullV2Labeled(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Nyu_depth_v2_labeled dataset."""

  VERSION = tfds.core.Version('2.1.0')
  RELEASE_NOTES = {
      # '1.0.0': 'Initial release.',
      # '2.0.0': 'different scenes for train/test',
      '2.1.0': 'additional "full" split',
  }

  # MANUAL_DOWNLOAD_INSTRUCTIONS = 1

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(Nyu_depth_v2_labeled): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
            'label': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint16),
        }),
        supervised_keys=("image", "label"),  # e.g. ('image', 'label')
        homepage='https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # (Nyu_depth_v2_labeled): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    download_dir = dl_manager.download(_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'dataset_path': download_dir,
                'scene_type': 'kitchen',
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'dataset_path': download_dir,
                'scene_type': 'bedroom',
            },
        ),
        tfds.core.SplitGenerator(
            name='full',
            gen_kwargs={
                'dataset_path': download_dir,
                'scene_type': None,
            },
        ),
    ]

  def _generate_examples(self, dataset_path, scene_type):
    """Yields examples."""
    # (Nyu_depth_v2_labeled): Yields (key, example) tuples from the dataset
    h5py = tfds.core.lazy_imports.h5py
    with h5py.File(dataset_path, 'r') as f:
      images = np.array(f['images'], dtype=f['images'].dtype).T.squeeze()
      labels = np.array(f['labels'], dtype=f['labels'].dtype).T.squeeze()
      scene_types = [
          f[f["sceneTypes"][0, i]][:, 0].tobytes().decode("utf-16")
          for i in range(f["sceneTypes"].shape[1])
      ]
      for i in range(images.shape[-1]):
        # Label_expand = np.expand_dims(labels[:,:,i], axis=2)
        if scene_type is None or scene_types[i] == scene_type:
          label = labels[:, :, i]
          yield str(i).zfill(4), {
              'image': images[:, :, :, i],
              'label': label.astype(np.uint16)
          }
