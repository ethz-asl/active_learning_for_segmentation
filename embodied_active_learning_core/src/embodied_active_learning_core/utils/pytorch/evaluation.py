import torch.utils.data as torchData
import os
import os.path
from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from densetorch.engine.miou import fast_cm

# This combines pillow + couch, aswell as counter + cabinet
COMBINE_PILLOW_COUCH_AND_COUNTER_CABINET = [
  [5, 17],
  [2, 11]
]


def semseg_accum_confusion_to_iou(confusion_accum):
  """
    Converts confsuion matrix to iou_mean and iou per class
  """
  conf = confusion_accum.double()
  diag = conf.diag()
  union = (conf.sum(dim=1) + conf.sum(dim=0) - diag)
  iou_per_class = 100 * diag / (union.clamp(min=1e-12))
  return iou_per_class


def score_model_for_dl(network: torch.nn.Module, loader: torch.utils.data.DataLoader, n_classes: int = 40,
                       classes_considered_for_miou: List[int] = None,
                       classes_to_combine: List[Optional[List[int]]] = []):
  """
    Calculates the miou, iou per class and accuracy for the given network

    network: Model to evaluate
    loader: Torch dataloader containing 'mask' and 'image'
    n_classes: Number of predicting classes
    classes_considered_for_miou: Over which classes miou should be calculated (None uses all classes in test set)
    classes_to_combine: Mapping of certain classes that should be combined (e.g. Counter + Cabinet, .. )

    returns: miou, iou for each class, accuracy
  """
  network = network.eval().cuda()
  confusion = np.zeros((n_classes, n_classes))
  gt_classes = np.asarray([])

  for batch in loader:
    imgs = batch['image'].cuda()
    masks = batch['mask'].cuda()
    h, w = masks.shape[-2:]
    with torch.no_grad():
      predictions = network(imgs)
      # If prediction is tuple, first value is semantic segmentation second is uncertainty
      if type(predictions) == tuple:
        predictions = predictions[0]

      predictions = F.interpolate(predictions, (h, w), mode="bilinear", align_corners=False)
      predictions_categorical = torch.argmax(predictions, dim=-3)

      idx = masks <= n_classes
      predictions_np = predictions_categorical[idx].cpu().detach().numpy().astype(np.uint8)
      gt_mask = masks[idx].cpu().detach().numpy().astype(np.uint8)

      for replacement_list in classes_to_combine:
        base_class = replacement_list[0]
        for c in replacement_list[1:]:
          predictions_np[predictions_np == c] = base_class
          gt_mask[gt_mask == c] = base_class

      confusion_matrix = fast_cm(
        predictions_np,
        gt_mask, n_classes
      )

      gt_classes = np.unique(np.append(gt_classes, gt_mask.ravel()))
      confusion += confusion_matrix

  if classes_considered_for_miou is None:
    classes_considered_for_miou = np.unique(gt_classes).astype(int)

  classIoU = semseg_accum_confusion_to_iou(torch.from_numpy(confusion))
  mIou = classIoU[classes_considered_for_miou].mean().cpu().detach().numpy().item()
  accuracy = np.sum(np.diag(confusion) / (np.sum(confusion)) * 100)

  return mIou, classIoU.cpu().detach().numpy(), accuracy.item()


class EarlyStoppingWrapper:
  """
    Class that implements early stopping
  """

  def __init__(self, model: torch.nn.Module, validation_dataloader: torch.utils.data.DataLoader,
               additional_dataloaders: List[torch.utils.data.DataLoader] = [],
               dataset_names: List[str] = ['Validation'], burn_in_steps: int = 5, training_step: int = 0):
    """
      Creates an early stopping wrapper. Use score_and_maybe_save() after each training iteration to score the model over all datasets and save it if the validation score is better

      model: torch network
      validation_dataloader: torch dataloader containing the validation data
      additional_dataloaders: list of additional dataloaders for which scores should be reported
      dataset_names: list of dataset names. Must have size additional_dataloaders + 1 (for validation set)
      burn_in_steps: Burn in steps. How many steps the validation score will be ignored. (Use this to make sure to only start early stopping after e.g. episode 5)
      training_step: The current training iteration of the network. (Used to write to logfile if requested)
    """
    self.best_miou = 0
    self.burn_in_steps = burn_in_steps
    self.model = model
    self.validation_dataloader = validation_dataloader
    self.additional_dataloaders = additional_dataloaders
    self.iteration = 0
    self.training_step = training_step

    for i in range(len(additional_dataloaders) + 1):
      if len(dataset_names) < i:
        dataset_names.append(f"Dataset_{i}")

    self.dataset_names = dataset_names

  def make_score_entry(self, mIoU: float, classes_iou: float, accuracy: float, dataset: str):
    """ Converts data to a dict that can then be saved using e.g. pandas """
    scores = {'mIoU': mIoU, 'Accuracy': accuracy, 'Dataset': dataset, 'Iteration': self.iteration,
              'Step': self.training_step}
    for class_idx, c in enumerate(classes_iou):
      scores[f'iou_class_{class_idx}'] = c
    return scores

  def score_and_maybe_save(self, verbose: bool = True, log_file: str = None, weights_file: str = None):
    """ Prints the scores of the network and if a better validation score is achieved, saves the weights
      verbose: Prints current scores
      weights_file: Where to save weights
      log_file: Where to save the log data (scores for all datasets). If file exists, data will be appended
    """
    all_scores = []
    miou, classes_miou, accuracy = score_model_for_dl(self.model, self.validation_dataloader)
    all_scores.append(self.make_score_entry(miou, classes_miou, accuracy, self.dataset_names[0]))

    if verbose:
      print("-" * 100)
      dataset_length = max([len(d) for d in self.dataset_names])  # For pretty printing
      color = '\33[32m' if miou >= self.best_miou else '\33[34m'
      print(
        color + f"[{self.dataset_names[0]:<{dataset_length}}] - Full mIou {np.mean(classes_miou):.2f}, mIoU {miou:.1f}%, Acc {accuracy:.2f}%" + '\33[0m')

    if self.iteration < self.burn_in_steps or miou >= self.best_miou:
      torch.save(self.model.state_dict(), weights_file)
      self.best_miou = miou

    # Print information about all other datasets.
    for ds_idx, dataset in enumerate(self.additional_dataloaders):
      miou, classes_miou, accuracy = score_model_for_dl(self.model, dataset)
      all_scores.append(self.make_score_entry(miou, classes_miou, accuracy, self.dataset_names[ds_idx + 1]))
      if verbose:
        print(
          f"[{self.dataset_names[ds_idx + 1]:<{dataset_length}}] - Full mIou {np.mean(classes_miou):.2f}, mIoU {miou:.1f}%, Acc {accuracy:.2f}%" + '\33[0m')

    self.iteration += 1

    if log_file is not None:
      pd.DataFrame.from_dict(all_scores).to_csv(log_file, mode='a', header=not os.path.exists(log_file))

    if verbose:
      print("-" * 100)
