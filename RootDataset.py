#!/usr/bin/env python3
import torch
import uproot
import numpy as np
import copy

# Based on https://github.com/jmduarte/capstone-particle-physics-domain/blob/master/weeks/GraphDataset.py

class RootDataset(torch.utils.data.Dataset):
  """RootDataset to handle ROOT files with pytorch
  Note:
    Removes unlabled events
  Members:
    root_file: Uproot file
    tree: uproot tree
    transform: transformation on data
    feature_array: Features for each event. [ (feat0_event0, feat1_event0, ...), (next event) ]
    spec_array: Spectators for each event.  [ (spec0_event0, spec1_event0, ...),  (next event) ]
    label_array: one-hot encoded label for each event. [[0, 1], [next event], ...]
  """
  def __init__(self, root_filename, tree_name, cut, features, spectators, class_branch, entry_stop=None, entry_start=None, normalize_min_max=None):
    """Initialization method
    Args:
      root_filename (str): Filename of ROOT file
      tree_name (str): Tree name in ROOT file
      cut (str): Cut on sample using uproot
      features (list): List of branches used for features
      spectators (list): List of branches used for spectators
      class_branch (str): Branch used for class label
      entry_stop (int): Event index used to set last entry
      entry_start (int): Event index used to set first entry
      normalize_min_max (list): Minimum and maximum used for normalizing features. If set to None, no normalization is applied.
    """
    self.root_file = uproot.open(root_filename)
    self.tree = self.root_file[tree_name]

    # feature_array = {feat1_name : [feat1_event0, feat1_event2, ...], }
    feature_array = self.tree.arrays(features,
                                cut,
                                library='np')

    # self.feature_array = [ (feat1_event0, feat2_event0, ...),  ]
    self.feature_array = np.stack([feature_array[feat][0] for feat in features], axis=1)
    spec_array = self.tree.arrays(spectators,
                             cut,
                             library='np')

    # self.spec_array = [ (spec1_event0, spec2_event0, ...),  ]
    self.spec_array = np.stack([spec_array[spec][0] for spec in spectators], axis=1)

    # label_array = {classId: [1, 1, 0, ... ]}
    label_array = self.tree.arrays(class_branch,
                              cut,
                              library='np')
    label_array = label_array[class_branch][0]
    # label_hotencoding = [ (0, 1), (0, 1), ... ]
    label_hotencoding = np.zeros((label_array.size, label_array.max()+1))
    label_hotencoding[np.arange(label_array.size), label_array] = 1
    self.label_array = np.array(label_hotencoding, dtype=int)

    # normalize
    if normalize_min_max:
      feat_min = np.amin(self.feature_array,0)
      feat_max = np.amax(self.feature_array,0)
      for ifeat, [min_x, max_x] in enumerate(normalize_min_max):
        self.feature_array[:,ifeat] = 2.*(self.feature_array[:,ifeat]-min_x)/(max_x-min_x) - 1.

    # Split data
    if entry_stop and entry_stop:
      self.feature_array = self.feature_array[entry_start:entry_stop]
      self.spec_array = self.spec_array[entry_start:entry_stop]
      self.label_array = self.label_array[entry_start:entry_stop]
    elif entry_stop:
      self.feature_array = self.feature_array[:entry_stop]
      self.spec_array = self.spec_array[:entry_stop]
      self.label_array = self.label_array[:entry_stop]
    elif entry_start:
      self.feature_array = self.feature_array[entry_start:]
      self.spec_array = self.spec_array[entry_start:]
      self.label_array = self.label_array[entry_start:]

    # remove unlabeled data
    self.feature_array = self.feature_array[np.sum(self.label_array, axis=1) == 1]
    self.spec_array = self.spec_array[np.sum(self.label_array, axis=1) == 1]
    self.label_array = self.label_array[np.sum(self.label_array, axis=1) == 1]


  def __len__(self):
    return len(self.label_array)

  def __getitem__(self, idx):
    sample = [self.feature_array[idx], self.label_array[idx], self.spec_array[idx]]
    return sample

def unnormalize(values, norm_min_max):
  """Unnormalizes features
  Args:
    values (list): Normalized features [ (feat0_event0, feat0_event0, ...), (next event) ]
    norm_min_max (list): Minimum and maximum used normalizing features [ (feat0_min, feat0_max), next feature ]
  Returns:
    feature_array (list); Unnormalized features [ (feat0_event0, feat0_event0, ...), (next event) ]
  """
  feature_array = copy.deepcopy(values)
  for ifeat, [min_x, max_x] in enumerate(norm_min_max):
    feature_array[:,ifeat] = (values[:,ifeat]+1)*(max_x-min_x)*1./2 + min_x
  return feature_array

if __name__ == '__main__':
  print("Example of using RootDataset")

  example_filename = 'data/example_data.root'
  features = ['min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 
              'llg_mass_err', 'phi', 
              'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
              'llg_flavor', 'llg_ptt', 
              'photon_pt_mass', 'lead_lep_pt', 'sublead_lep_pt']
  normalize_max_min = [[0.4,3.7], [0.4,5.0], [0.0,11.0], [-1.0, 1.0], [-1.0, 1.0],
                      [1.1, 3.4], [0.0, 6.3],
                      [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                      [0., 1.], [0.0, 400.], 
                      [0.13, 2.0], [25.0, 1000.], [15.0, 500.]]
  example_dataset = RootDataset(root_filename= example_filename,
                            tree_name = "eval_full_tree",
                            features = features,
                            normalize_min_max = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass'],
                            class_branch = 'classID')

  print(f'Entries: {len(example_dataset)}')
  print(f'Features: {features}')
  print(f'Feature values: {unnormalize(example_dataset.feature_array, normalize_max_min)}')
  print(f'Normalized feature values: {example_dataset.feature_array}')
  print(f'Labels: {example_dataset.label_array[:,1]}')
  print(f'Spectators: {example_dataset.spec_array}')
