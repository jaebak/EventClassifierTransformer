#!/usr/bin/env python3
from torch import nn
import torch
import RootDataset
import uproot
import array
import ROOT
import ctypes
import math
import os

def infer_sample(dataloader, model, loss_fn, mass_index, weight_index):
  """Infers network
  Args:
    dataloader: torch.utils.data.DataLoader
    model: network
    loss_fn: loss function used in results
    mass_index: index of observable in spectator of RootDataset
    weight_index: index of weight in spectator of RootDataset
  Returns:
    results (dict): {loss: [], x: [], y: [], yhat: [], mass: [], weight: []}
  """
  # Setup
  nBatches = len(dataloader)
  loss = 0.
  model.eval()
  x_array, y_array, pred_array, mass_array, weight_array = [], [], [], [], []
  # Collect information for evaluation
  with torch.no_grad():
    # Evaluate sample
    for feature, label, spec in dataloader:
      # feature: (batch, n_features)
      # label: (batch, 2)
      # spec: (batch, n_spectators)

      # Setup
      X, y = feature, torch.max(label,1)[1] # returns (batch, n_features), (batch)
      pred = model(X) # returns (batch, 1)
      loss += loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32)) # returns (batch)
      # Add results to arrays
      x_array.extend(X.cpu().numpy())
      y_array.extend(y.cpu().numpy())
      pred_array.extend(pred.squeeze().cpu().numpy())
      mass_array.extend(spec[:,mass_index])
      weight_array.extend(spec[:,weight_index])
  # Post batch loop operations
  loss /= nBatches
  # Save results
  results = {}
  results['loss'] = loss
  results['x'] = x_array
  results['y'] = y_array
  results['yhat'] = pred_array
  results['mass'] = mass_array
  results['weight'] = weight_array
  return results

def find_sample_fraction_thresholds(sample_fractions, tmva_chain, var_name, sample_cut, weight_name = '', include_min_max=False):
  """Finds thresholds that divides sample into bins of var_name based on fractions in sample_fractions
  Args:
    sample_fractions (list): List of fractions to divide sample
    tmva_chain (ROOT.TTree): sample in TTree format
    var_name (str): variable used to split bins
    sample_cut (str): cut applied before dividing sample
    weight_name (str): weight name for sample
    include_min_max (bool): include max and min threshold of var_name in returned var_thresholds
  Returns:
    var_thresholds (list): thresholds of var_name that divides sample according to sample_factions
  """
  # Find min max of variable
  min_val = tmva_chain.GetMinimum(var_name)
  max_val = tmva_chain.GetMaximum(var_name)
  # Make histogram of var_name
  hist_var = ROOT.TH1F("hist_var","hist_var",10000,min_val*0.9,max_val*1.1) # Make histogram larger than min and max
  if weight_name == '': n_entries = tmva_chain.Draw(f'{var_name}>>hist_var', sample_cut, 'goff')
  else: n_entries = tmva_chain.Draw(f'{var_name}>>hist_var', f'({sample_cut})*{weight_name}', 'goff')
  # Split histogram into quantiles
  var_quantiles = array.array('d', [0.]*len(sample_fractions))
  var_fractions = array.array('d', [1.-sample_fraction for sample_fraction in sample_fractions])
  hist_var.GetQuantiles(len(sample_fractions), var_quantiles, var_fractions)
  var_thresholds = var_quantiles.tolist()
  # Includes min and max if needed
  if include_min_max:
    var_thresholds.insert(0,min_val)
    var_thresholds.append(max_val)
  return var_thresholds

def calculate_significance(root_filename, tree_name, y_name, y_signal, y_background, yhat_name, weight_name, observable_name, observable_hist_def, fixed_width=False, detail_output=False):
  """Calculates combined significance when splitting sample into bins of yhat_name
  Args:
    root_filename (str): sample in ROOT format
    tree_name (str): tree name in ROOT file
    y_name (str): Label name in tree
    y_signal (int): Label value that indicates signal
    y_background (int): Label value that indicates background
    yhat_name (str): MVA discriminator name in tree
    weight_name (str): weight name in tree
    observable_name (str): observable name in tree. Ex) mass.
    observable_hist_def (list): [n_bins, min, max]
    fixed_width (bool): Use a fixed width (set to 120 to 130 GeV) of the observable when calculating significance. 
      Otherwise measure signal resolution to determine window of observable to use when calculating significance that depends on each bin.
    detail_output (bool): Return information for plotting
  Returns:
    detail_output = False: integrated_significance (float), integrated_significance_err (float)
    detail_output = True:  integrated_significance (float), integrated_significance_err (float), 
                             [integrated_significance (float), integrated_significance_err (float), signal_fraction_bins (list), significances (list), significance_errs (list), signal_widths (list)]
  """
  # Get equal signal efficiency bins divided by MVA discriminator
  nbins = 8
  signal_fraction_edges = [1-(ibin+1)/(nbins) for ibin in range(nbins-1)] #[0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)
  mva_threshold_edges = find_sample_fraction_thresholds(signal_fraction_edges, tmva_chain, yhat_name, f'{y_name}=={y_signal}', weight_name, include_min_max=True)

  # Storage for detail plots
  # signal_widths = [ [90%(low),65%(low),65%(high),90%(high)] for each MVA bin]
  signal_widths = []
  # significances = [ significance for each MVA bin]
  significances = []
  # significance_errs = [ significance error for each MVA bin]
  significance_errs = []
  # signal_fraction_bins = [mean fraction of sample for each MVA bin]
  signal_fraction_bins = []

  # Calculate significance for each mva bin
  hist = ROOT.TH1F("hist","hist",observable_hist_def[0],observable_hist_def[1],observable_hist_def[2])
  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    if ithresh == 0: continue
    # Make mva window cut
    if ithresh == 1: mva_window = f'{yhat_name}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{yhat_name}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{yhat_name}<{mva_threshold}&&{yhat_name}>{mva_threshold_edges[ithresh-1]}'
    entries = tmva_chain.Draw(f'{observable_name}>>hist',f'({y_name}==1&&{mva_window})*{weight_name}',"goff")
    # Calculate signal fraction
    if ithresh == len(mva_threshold_edges)-1: signal_fraction_bins.append(1./len(signal_fraction_edges)/2)
    else: signal_fraction_bins.append(signal_fraction_edges[ithresh-1]+1./len(signal_fraction_edges)/2)
    # Find signal width
    if fixed_width: signal_widths.append([120, 120, 130, 130])
    else:
      mva_signal_width = hist.GetStdDev()
      mass_fractions = [0.95, 0.84, 0.16, 0.05] # 90%, 65%
      mass_quantiles = array.array('d', [0.]*len(mass_fractions))
      mass_fractions = array.array('d', [1.-mass_fraction for mass_fraction in mass_fractions])
      hist.GetQuantiles(len(mass_fractions), mass_quantiles, mass_fractions)
      mass_thresholds = mass_quantiles.tolist()
      signal_widths.append(mass_thresholds)

    # Find signal yield within 2 sigma of signal width
    mass_window = f'{observable_name}<{signal_widths[-1][3]}&&{observable_name}>{signal_widths[-1][0]}' #90% of signal
    tmva_chain.Draw(f"{observable_name}>>hist",f'({y_name}=={y_signal}&&{mva_window}&&{mass_window})*{weight_name}',"goff")
    nentries_signal = hist.GetEntries()
    bin_s_err = ctypes.c_double()
    nevents_signal = hist.IntegralAndError(0,hist.GetNbinsX()+1, bin_s_err);
    bin_s_err = bin_s_err.value
    # Find background yield within 2 sigma of signal width
    tmva_chain.Draw(f"{observable_name}>>hist",f'({y_name}=={y_background}&&{mva_window}&&{mass_window})*{weight_name}',"goff")
    nentries_background = hist.GetEntries()
    bin_b_err = ctypes.c_double()
    nevents_background = hist.IntegralAndError(0,hist.GetNbinsX()+1, bin_b_err);
    bin_b_err = bin_b_err.value

    # Append MVA bin significance and significance error
    if nevents_background == 0: 
      significances.append(0)
      significance_errs.append(0)
    else: 
      significances.append(math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal)))
      # Error propagation based on s/sqrt(b) for simplicty
      significance_errs.append(math.sqrt(1/(significances[-1]**2)*((nevents_signal/nevents_background)**2)*(bin_s_err**2) + 1/(significances[-1]**2)/4*((nevents_signal/nevents_background)**4)*(bin_b_err**2)))

  # Calculate integrated significance
  integrated_significance = 0.
  for significance in significances:
    integrated_significance += significance**2
  integrated_significance = math.sqrt(integrated_significance)
  integrated_significance_err = 0.
  for significance_err in significance_errs:
    integrated_significance_err += significance_err**2
  integrated_significance_err = math.sqrt(integrated_significance_err)

  if detail_output: return integrated_significance, integrated_significance_err, [integrated_significance, integrated_significance_err, signal_fraction_bins, significances, significance_errs, signal_widths]
  else: return integrated_significance, integrated_significance_err

if __name__ == '__main__':
  print("Example of training with significance based model selection")

  # Set seed for constant results
  torch.manual_seed(1)

  # Setup sample
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
                            spectators = ['llg_mass','w_lumiXyear'],
                            class_branch = 'classID')
  mass_index = 0
  weight_index = 1
  train_dataloader = torch.utils.data.DataLoader(example_dataset, batch_size=100, shuffle=False)

  # Setup model
  n_features = len(features)
  model = nn.Sequential(nn.Linear(n_features, 3*n_features), nn.Tanh(), nn.Linear(3*n_features, 1), nn.Sigmoid())

  # Setup loss and optimizer
  loss_fn = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Make output directory
  if not os.path.exists('output'):
    os.makedirs('output')

  # Train network for two epoch, where network with highest significance is selected.
  print('Training network')
  best_significance = 0
  for iEpoch in range(4):
    print(f'Epoch {iEpoch}')
    model.train()
    avg_loss = 0.
    # Train a batch
    for batch, (feature, label, spec) in enumerate(train_dataloader):
      # Setup
      X, y = feature, torch.max(label,1)[1].to(torch.float32)
      pred = model(X)
      mass = spec[:,0].unsqueeze(1)
      # Calculate loss
      loss = loss_fn(pred.squeeze(), y)
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
    # Evaluate epoch
    results = infer_sample(train_dataloader, model, loss_fn, mass_index=mass_index, weight_index=weight_index)
    infered_results_filename = 'output/infered_results.root'
    infered_results_tree_name = 'eval_full_tree'
    # Write results
    with uproot.recreate(infered_results_filename) as root_file:
        root_file[infered_results_tree_name] = {'x': results['x'], 'y': results['y'], 'yhat': results['yhat'], 'mass': results['mass'], 'weight': results['weight']}
    print('  Wrote infered results to '+infered_results_filename)
    integrated_significance, integrated_significance_err, signi_detail = calculate_significance(infered_results_filename, infered_results_tree_name, y_name='y', y_signal=1, y_background=0, yhat_name='yhat', weight_name='weight', observable_name='mass', observable_hist_def=[160,100,180], fixed_width=False, detail_output=True)
    print(f'  Significance: {integrated_significance:.3f} +- {integrated_significance_err:.3f}')
    # Save model if significance is highest
    if best_significance < integrated_significance:
      best_significance = integrated_significance
      print(f'  Significance was improved for {iEpoch} epoch.')
      print(f'    Saving model as "output/best_network.pt"')
      torch.save(model.state_dict(), 'output/best_network.pt')
