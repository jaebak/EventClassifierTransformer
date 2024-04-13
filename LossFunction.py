#!/usr/bin/env python3
from torch import nn
import torch
import RootDataset
import Disco

class extreme_disco_loss(nn.Module):
  """Extreme disco loss within data scope 
  Members:
    disco_factor (int): Disco factor
    data_scope_min (float): Minimum value for scope of data
    data_scope_max (float): Maximum value for scope of data
    clamp_min (float): Minimum value that network output is clampped to. Helps preventing network diverge. 
    clamp_max (float): Minimum value that network output is clampped to. Helps preventing network diverge. 
    background_label (int): Background label used for selecting events to be in Disco calculation
  """
  def __init__(self, disco_factor = 10., data_scope_min=125.-5, data_scope_max=125.+5, clamp_min=0.001, clamp_max=0.999, background_label=0):
    super(extreme_disco_loss, self).__init__()
    self.disco_factor = disco_factor
    self.data_scope_min = data_scope_min
    self.data_scope_max = data_scope_max
    self.clamp_min = clamp_min
    self.clamp_max = clamp_max
    self.background_label = background_label

  def forward(self, output, target, mass):
    """Calculation of loss
    Args:
      output (torch.tensor(batch, 1)): output of network
      target (torch.tensor(batch, 1)): label
      mass (torch.tensor(batch, 1)): observable used for data scope training
    Returns:
      loss (torch.tensor(1)): extreme disco loss of network
    """
    # Squeezed variables for data scope and disco
    mass_list = mass.squeeze()
    output_list = output.squeeze()

    # For classifier loss which uses events in data scope
    with torch.no_grad():
      mass_mask = ((mass_list>self.data_scope_min)&(mass_list<self.data_scope_max)).unsqueeze(1) # returns (batch, 1)
    output_window = output[mass_mask].unsqueeze(1) # returns (batch within data scope, 1)
    target_window = target[mass_mask].unsqueeze(1) # returns (batch within data scope, 1)
    clamp_output = torch.clamp(output_window, min=0.001, max=0.999)
    # Calculate loss
    extreme = (target_window-1.) / (clamp_output-1.) + target_window / clamp_output + (2*target_window-1.)*torch.log(1.-clamp_output) + (1.-2*target_window)*torch.log(clamp_output) # returns (batch within data scope, 1)
    # Could apply event weights at this point.
    extreme_loss = torch.mean(extreme) # returns (1)

    # For disco which uses background events
    normedweight = torch.tensor([1.]*len(mass)) # returns (batch)
    # Apply mask that selects only background events
    with torch.no_grad():
      mask = (target.squeeze() == self.background_label)
    mass_list_bkg = mass_list[mask] # returns (batch)
    output_list_bkg = output_list[mask] # returns (batch)
    normedweight_bkg = normedweight[mask] # returns (batch)
    disco = Disco.distance_corr(mass_list_bkg, output_list_bkg, normedweight_bkg) # returns (1)

    # Loss calculation
    loss = extreme_loss + self.disco_factor * disco # returns (1)
    return loss

class bce_disco_loss(nn.Module):
  """BCE disco loss within data scope 
  Members:
    disco_factor (int): Disco factor
    data_scope_min (float): Minimum value for scope of data
    data_scope_max (float): Maximum value for scope of data
    background_label (int): Background label used for selecting events to be in Disco calculation
    bce_loss : BCE loss
  """
  def __init__(self, disco_factor = 10., data_scope_min=125.-5, data_scope_max=125.+5, background_label=0):
    super(bce_disco_loss, self).__init__()
    self.bce_loss = nn.BCELoss(reduction='none')
    self.disco_factor = disco_factor
    self.data_scope_min = data_scope_min
    self.data_scope_max = data_scope_max
    self.background_label = background_label

  def forward(self, output, target, mass):
    """Calculation of loss
    Args:
      output (torch.tensor(batch, 1)): output of network
      target (torch.tensor(batch, 1)): label
      mass (torch.tensor(batch, 1)): observable used for data scope training
    Returns:
      loss (torch.tensor(1)): extreme disco loss of network
    """
    # Squeezed variables for data scope and disco
    mass_list = mass.squeeze()
    output_list = output.squeeze()

    # For classifier loss, use events in mass window
    with torch.no_grad():
      mass_mask = ((mass_list>self.data_scope_min)&(mass_list<self.data_scope_max)).unsqueeze(1) # returns (batch, 1)
    output_window = output[mass_mask].unsqueeze(1) # returns (batch within data scope, 1)
    target_window = target[mass_mask].unsqueeze(1) # returns (batch within data scope, 1)
    # Calculate loss
    bce_window = self.bce_loss(output_window, target_window) # returns (batch within data scope, 1)
    # Could apply event weights at this point.
    bce_loss = torch.mean(bce_window) # returns (1)

    # For disco which uses background events
    normedweight = torch.tensor([1.]*len(mass)) # returns (batch)
    # Apply mask that selects only background events
    with torch.no_grad():
      mask = (target.squeeze() == self.background_label) 
    mass_list_bkg = mass_list[mask] # returns (batch)
    output_list_bkg = output_list[mask] # returns (batch)
    normedweight_bkg = normedweight[mask] # returns (batch)
    disco = Disco.distance_corr(mass_list_bkg, output_list_bkg, normedweight_bkg) # returns (1)

    # Loss calculation
    loss = bce_loss + self.disco_factor * disco # returns (1)
    return loss

if __name__ == '__main__':
  print("Example of training with extreme-disco function")

  # Set seed for constant results
  torch.manual_seed(1)

  # Load data
  features = ['min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 
              'llg_mass_err', 'phi', 
              'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
              'llg_flavor', 'llg_ptt', 
              'photon_pt_mass', 'lead_lep_pt', 'sublead_lep_pt']
  normalize_min_max = [[0.4,3.7], [0.4,5.0], [0.0,11.0], [-1.0, 1.0], [-1.0, 1.0],
                      [1.1, 3.4], [0.0, 6.3],
                      [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                      [0., 1.], [0.0, 400.], 
                      [0.13, 2.0], [25.0, 1000.], [15.0, 500.]]
  example_filename = 'data/example_data.root'
  example_dataset = RootDataset.RootDataset(root_filename= example_filename,
                            tree_name = "eval_full_tree",
                            features = features,
                            normalize_min_max = normalize_min_max,
                            cut = '1',
                            spectators = ['llg_mass'],
                            class_branch = 'classID')
  mass_spectator_index = 0

  # Make model
  n_features = len(features)
  model = nn.Sequential(nn.Linear(n_features, 3*n_features), nn.Tanh(), nn.Linear(3*n_features, 1), nn.Sigmoid())

  # Setup training
  train_dataloader = torch.utils.data.DataLoader(example_dataset, batch_size=100, shuffle=False)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Train for two epoch using extreme-disco loss
  print('Training using extreme disco loss')
  loss_fn = extreme_disco_loss()
  for iEpoch in range(2):
    model.train()
    avg_loss = 0.
    # Train a batch
    for batch, (feature, label, spec) in enumerate(train_dataloader):
      # feature: (batch, n_features)
      # label: (batch, 2)
      # spec: (batch, n_spectators)

      # Setup
      X, y = feature, torch.max(label,1)[1] # returns (batch, n_features), (batch)
      pred = model(X) # returns (batch, 1)
      mass = spec[:,mass_spectator_index].unsqueeze(1) # returns (batch, 1)
      # Calculate loss
      loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32), mass.to(torch.float32)) # returns (1)
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # For printing average loss
      avg_loss += loss.item() * len(X) # average loss over batch
    # Print average loss for epoch
    size = len(train_dataloader.dataset)
    avg_loss = avg_loss / size
    print(f'  avg loss for epoch {iEpoch}: {avg_loss:>7f}')

  # Train for one epoch using bce-disco loss
  print('Training using BCE disco loss')
  loss_fn = bce_disco_loss()
  for iEpoch in range(2):
    model.train()
    avg_loss = 0.
    # Train a batch
    for batch, (feature, label, spec) in enumerate(train_dataloader):
      # feature: (batch, n_features)
      # label: (batch, 2)
      # spec: (batch, n_spectators)

      # Setup
      X, y = feature, torch.max(label,1)[1] # returns (batch, n_features), (batch)
      pred = model(X) # returns (batch, 1)
      mass = spec[:,mass_spectator_index].unsqueeze(1) # returns (batch, 1)
      # Calculate loss
      loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).type(torch.float32), mass.to(torch.float32)) # returns (1)
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # For printing average loss
      avg_loss += loss.item() * len(X) # loss is average over batch
    # Print average loss for epoch
    size = len(train_dataloader.dataset)
    avg_loss = avg_loss / size
    print(f'  avg loss for epoch {iEpoch}: {avg_loss:>7f}')
