import torch
import torch.nn as nn


class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=False, batch_norm=False):
        super(TwoLayerNN, self).__init__()
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.layer_1 = nn.Linear(input_size, hidden_size) 
        self.layer_out = nn.Linear(hidden_size, output_size)
        
        if dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Identity()
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)
        else:
            self.bn = nn.Identity()
        
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.layer_out(x)
            
        return x
    
    def del_neuron(self, index):
        # take a copy of the current weights of layer_out
        current_layer_out = [i.data for i in self.layer_out.weight]
        current_layer_out = torch.stack(current_layer_out)
        
        # cut input neuron for the second layer
        layer_out_new_weight = torch.cat([current_layer_out[:, 0:index], current_layer_out[:, index+1:]], dim=1)
        self.layer_out = nn.Linear(layer_out_new_weight.shape[1], self.output_size)
        self.layer_out.weight.data = layer_out_new_weight.clone().detach().requires_grad_(True)
        
        # cut output neuron for the first layer
        current_layer_1 = [i.data for i in self.layer_1.weight]
        current_layer_1 = torch.stack(current_layer_1)
        
        layer_1_new_weight = torch.cat([current_layer_1[:index], current_layer_1[index+1:]])
        self.layer_1 = nn.Linear(layer_1_new_weight.shape[1], layer_1_new_weight.shape[0])
        self.layer_1.weight.data = layer_1_new_weight.clone().detach().requires_grad_(True)
        
        # make sure the shape of Batch Norm layer
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(layer_1_new_weight.shape[0])
    
    # def add_neuron(self, n_neuron, layer):
    #     current = [i.data for i in layer.weight]
    #     current = torch.stack(current)
        
    #     # initialize tensor with wanted size
    #     hl_input = torch.zeros([current.shape[0], 1])
    #     nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain('relu'))