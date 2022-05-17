import torch
import torch.nn as nn
import torch.optim as optim

from model import TwoLayerNN, LinearRegression


class InitModel:
    def __init__(self, input_size, hidden_size, output_size, device, dropout=False, batch_norm=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.model = TwoLayerNN(input_size, hidden_size, output_size, dropout=dropout, batch_norm=batch_norm)
    
    def init_module_1_ReLU_LR(self, train_loader):
        if self.hidden_size != 1:
            raise BaseException('hidden_size should be 1')
        
        linear_model, loss = self._train_linear(train_loader)
        new_weight = linear_model.linear.weight.data
        new_layer_bias = linear_model.linear.bias.data
        new_output_bias = torch.tensor([[loss]], dtype=torch.float32)
        
        self.model.layer_1.weight.data = new_weight
        self.model.layer_1.bias.data = new_layer_bias
        self.model.layer_out.weight.data = torch.tensor([[1]], dtype=torch.float32)
        self.model.layer_out.bias.data = new_output_bias
        
        return self.model.to(self.device)
    
    def init_module_multi_ReLU_WT(self):
        '''initialize random weight'''
        self.model.layer_1.weight.data.normal_(0, 1)
        self.model.layer_1.bias.data.zero_()
        self.model.layer_out.weight.data.normal_()
        self.model.layer_out.bias.data.zero_()
        
        return self.model.to(self.device)
        
    def _train_linear(self, train_loader):
        
        model = LinearRegression(12, 1).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for _ in range(100):
                
            model.train()
            train_loss = []
            
            for batch in train_loader:
                
                x, y = batch
                
                optimizer.zero_grad()
                y_pred = model(x.to(self.device))
                
                loss = criterion(y_pred.squeeze(), y.to(self.device))
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
            
            train_loss = sum(train_loss) / len(train_loss)
        
        return model, train_loss