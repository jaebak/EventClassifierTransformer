#!/usr/bin/env python3
from torch import nn
import torch
import RootDataset

class EventClassifierTransformerNetwork(nn.Module):
  """Event Classifier Transformer
  Notes:
    Each feature becomes a token, with feature dependent feed-forward network. 
    Class token is used to reduce dimensions.
  Members:
   embed: Feature dependent feed-forward layer to embed features
   pre_attn_norm: [encoder] Normalization layer used before multihead attention layer
   attn: [encoder] Multihead attenion layer
   pre_fc_norm: [encoder] Normalization layer used before position-wise network layer
   fc1, act, fc2: [encoder] Position-wise network layer
   cls_token: [decoder] Class token
   cls_pre_attn_norm: [decoder] Normalization layer used before class multihead attention layer
   cls_attn: [decoder] Multihead attenion layer
   cls_pre_fc_norm: [decoder] Normalization layer used before linear layer
   cls_fc: [decoder] Linear layer
   sigmoid: Activation function for last layer
   dropout: Dropout layer used in Multihead attention, feed-forward layers
  """
  def __init__(self, nvars, dropout=0.1, nodes=16, nheads=4, linear_factor=4):
    """Initialization method
    Args:
      nvars (int): Number of features
      dropout (float): dropout factor
      nodes (int): Number of nodes for embedding feature
      linear_factor (int): Number used for increasing dimentions in linear layers
      nheads (int): Number of heads used in multihead attention. nodes/nheads should be an integer
    """
    super().__init__()
    self.embed = nn.ModuleList([nn.Sequential(nn.Linear(1,linear_factor), nn.GELU(), nn.Linear(linear_factor,nodes), nn.GELU()) for _ in range(nvars)])
    self.pre_attn_norm = nn.LayerNorm(nodes)
    self.attn = nn.MultiheadAttention(embed_dim=nodes,num_heads=nheads,dropout=dropout)
    self.pre_fc_norm = nn.LayerNorm(nodes)
    self.fc1 = nn.Linear(nodes, linear_factor*nodes)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(linear_factor*nodes, nodes)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, nodes), requires_grad=True)
    self.cls_pre_attn_norm = nn.LayerNorm(nodes)
    self.cls_attn = nn.MultiheadAttention(embed_dim=nodes,num_heads=nheads,dropout=dropout)
    self.cls_pre_fc_norm = nn.LayerNorm(nodes)
    self.cls_fc = nn.Linear(nodes, 1)
    self.sigmoid = torch.nn.Sigmoid()
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x):
    """Prediction of network
    Args:
      x (torch.tensor(batch, nfeatures)): features for network
    Returns:
      yhat (torch.tensor(batch, 1)): prediction of network
    """
    # Embed layer
    var_embed_list = []
    for ivar, var_embed_nn in enumerate(self.embed):
        var_embed_list.append(var_embed_nn(x[:,ivar].unsqueeze(1)))
    var_embed = torch.stack(var_embed_list,axis=1) # returns (batch, nvars, nodes)
    
    # Attention layer
    x = var_embed.permute(1,0,2).contiguous() # returns (nvars, batch, nodes)
    residual = x
    x = self.pre_attn_norm(x)
    x = self.attn(x, x, x)[0]
    x = self.dropout(x)
    x += residual # returns (nvars, batch, nodes)
    
    # FC layer
    residual = x
    x = self.pre_fc_norm(x)
    x = self.act(self.fc1(x))
    x = self.fc2(x)
    x = self.dropout(x)
    x += residual # returns (nvars, batch, nodes)

    # Class layer
    cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # returns (1, batch, nodes)
    
    # Attention layer
    residual = cls_tokens
    u = torch.cat((cls_tokens, x), dim=0) # returns (nvars+1, batch, nodes)
    u = self.cls_pre_attn_norm(u)
    x = self.cls_attn(cls_tokens, u, u)[0] # returns (1, batch, nodes)
    x = self.dropout(x)
    x += residual # returns (1, batch, nodes)
    
    # FC layer
    x = self.cls_pre_fc_norm(x)
    x = x.permute(1,0,2) # returns (batch, 1, nodes)
    x = x.flatten(start_dim=1) # returns (batch, nodes)
    x = self.cls_fc(x) # returns (batch, 1)

    out = self.sigmoid(x) # returns (batch, 1)
    
    return out

if __name__ == '__main__':
  print("Example of using EventClassifierTransformer")

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

  # Make model
  model = EventClassifierTransformerNetwork(nvars=len(features),dropout=0.1)
  # Load model weights
  model_weights = 'model_weights/EventClassifierTransformer_weights.pt'
  state_dict = torch.load(model_weights, map_location=torch.device('cpu'))
  model.load_state_dict(state_dict)

  # Evaluate model
  model.eval()
  with torch.no_grad():
    predict = model(torch.from_numpy(example_dataset.feature_array)).squeeze().numpy()

  # Print 
  print(f'Number of data entries: {len(example_dataset)}')
  print(f'Features: {features}')
  print(f'Feature values: {RootDataset.unnormalize(example_dataset.feature_array, normalize_min_max)}')
  print(f'Normalized feature values: {example_dataset.feature_array}')
  print(f'Labels (signal is 1): {example_dataset.label_array[:,1]}')
  print(f'Prediction (signal is 1): {predict}')
