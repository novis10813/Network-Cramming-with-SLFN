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
        x = torch.relu(self.layer_1(inputs))
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
    
    def add_neuron(self, train_loader, index):
        # cut output neuron weight for the first layer
        current_weight_1 = self.layer_1.weight.data
        current_bias_1 = self.layer_1.bias.data
        
        new_neuron_weight = train_loader.dataset[index][0].unsqueeze(0).detach().cpu()
        new_neuron_bias = torch.tensor([1 - len(train_loader.dataset[0][index])], dtype=torch.float32)
        
        new_weights_1 = torch.cat([current_weight_1, new_neuron_weight])
        new_bias_1 = torch.cat([current_bias_1, new_neuron_bias])
        
        self.layer_1 = nn.Linear(new_weights_1.shape[1], new_weights_1.shape[0])
        self.layer_1.weight.data = new_weights_1.clone().detach().requires_grad_(True)
        self.layer_1.bias.data = new_bias_1.clone().detach().requires_grad_(True)
        
        # take a copy of the current weights of layer_out
        current_weight_out = self.layer_out.weight.data
        current_bias_out = self.layer_out.bias.data
        
        y = train_loader.dataset[index][1].unsqueeze(0).detach().cpu()
        new_neuron_out = y - current_bias_out - torch.relu(current_weight_out).sum().detach().cpu()
        
        new_weights_out = torch.cat([current_weight_out, new_neuron_out])
        self.layer_out = nn.Linear(new_weights_out.shape[1], new_weights_out.shape[0])
        self.layer_out.weight.data = new_weights_out.clone().detach().requires_grad_(True)
        self.layer_out.bias.data = current_bias_out.clone().detach().requires_grad_(True)
        
        # make sure the shape of Batch Norm layer
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(new_weights_1.shape[0])


class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        
        # define layers
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
    
        return self.linear(x)


class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, dropout=False, batch_norm=False):
        super(Decoder, self).__init__()
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
        x = torch.relu(self.layer_1(inputs))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.layer_out(x)
            
        return x


class Autoencoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, dropout=False, batch_norm=False):
        super(Autoencoder, self).__init__()
        self.encoder = TwoLayerNN(input_size, hidden_size, output_size, dropout=dropout, batch_norm=batch_norm)
        self.decoder = Decoder(input_size=output_size, hidden_size=hidden_size, output_size=input_size, dropout=dropout, batch_norm=batch_norm)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x