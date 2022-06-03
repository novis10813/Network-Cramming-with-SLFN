import copy
import torch
import itertools

from DataPreprocess import LTS_dataloader
from utils import print_info
from model import TwoLayerNN

def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_result_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_result_sum / y_true.shape[0]
    
    return acc


# Basic training pipeline
def train(train_loader=None, val_loader=None, model=None, epochs=None, criterion=None, optimizer=None, l1_lambda=None, l2_lambda=None, binary=True):
    
    for epoch in range(epochs):
        
        model.train()
        train_loss = []
        train_accs = []
        
        for batch in train_loader:
            
            x, y = batch
            
            logits = model(x)
            
            if binary:
                loss = criterion(logits, y)
                acc = binary_acc(logits, y)
            else:
                loss = criterion(logits, y)
                acc = (logits.argmax(dim=-1) == y).float().mean()
            
            # L1 regularization with normalized l1
            if l1_lambda is not None:
                L1_regularization = sum(p.abs().sum() for p in model.parameters())
                param_num = sum(p.numel() for p in model.parameters())
                loss += (l1_lambda / param_num) * L1_regularization
                        
            # L2 regularization with normalized l2
            if l2_lambda is not None:
                L2_regularization = sum(p.pow(2.0).sum() for p in model.parameters())
                param_num = sum(p.numel() for p in model.parameters())
                loss += (l2_lambda / param_num) * L2_regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        model.eval()

        valid_loss = []
        valid_accs = []
        
        for batch in val_loader:
            x, y = batch
            
            with torch.no_grad():
                logits = model(x)
                
                if binary:
                    acc = binary_acc(logits, y)
                else:
                    acc = (logits.argmax(dim=-1) == y).float().mean()
                
                valid_loss.append(loss.item())
                valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
    
    return model


def multiclass_regularization(train_loader=None,
                              val_loader=None,
                              epochs:int=None,
                              model:torch.nn.Module=None,
                              optimizer:torch.optim=None,
                              criterion=None,
                              l1_lambda:float=None,
                              l2_lambda:float=None,
                              loss_threshold:float=None,
                              eta_threshold:float=None,
                              device=None,
                              binary:bool=True):
    '''
    Args:
    epochs: If None, it will not stop by the limitation of epoch, otherwise the training process will stop at your desired epoch.
    model: A neural network.
    optimizer: Training optimizer.
    loss_threshold: stopping criteria for training loss.
    eta_threshold: stopping criteria for learning rate.
    l1_lambda: if None, it won't apply l1 regularization to the model.
    l2_lambda: if None, it won't apply l2 regularization to the model.  
    '''
    
    previous_train_loss = 10000
    try:
        for epoch in itertools.count():
            
            model.train().to(device)
            
            previous_model_params = model.state_dict()
            stop_training = False
            
            while True:
                
                train_loss = []
                train_accs = []
                
                for batch in train_loader:
                    
                    x, y = batch
                    
                    logits = model(x.to(device))
                    
                    if binary:
                        loss = criterion(logits, y.to(device))
                    else:
                        loss = criterion(logits, y.to(device))
                    
                    # L1 regularization with normalized l1
                    if l1_lambda is not None:
                        L1_regularization = sum(p.abs().sum() for p in model.parameters())
                        param_num = sum(p.numel() for p in model.parameters())
                        loss += (l1_lambda / param_num) * L1_regularization
                    
                    # L2 regularization with normalized l2
                    if l2_lambda is not None:
                        L2_regularization = sum(p.pow(2.0).sum() for p in model.parameters())
                        param_num = sum(p.numel() for p in model.parameters())
                        loss += (l2_lambda / param_num) * L2_regularization
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if binary:
                        acc = binary_acc(logits, y)
                    else:
                        acc = (logits.argmax(dim=-1) == y).float().mean()
                    
                    train_loss.append(loss.item())
                    train_accs.append(acc)
                
                max_train_loss = max(train_loss)
                train_loss = sum(train_loss) / len(train_loss)
                train_acc = sum(train_accs) / len(train_accs)
                
                if eta_threshold is not None:
                    if train_loss <= previous_train_loss:
                        if max_train_loss < loss_threshold:
                            optimizer.param_groups[0]['lr'] *= 1.2
                            previous_train_loss = train_loss
                            break
                        
                        else:
                            model.load_state_dict(previous_model_params)
                            stop_training = True
                            # print(f'max loss:{max_train_loss} > threshold{loss_threshold}, stop training.')
                            break
                    
                    if optimizer.param_groups[0]['lr'] > eta_threshold:
                        optimizer.param_groups[0]['lr'] *= 0.8
                        model.load_state_dict(previous_model_params)
                    
                    else:
                        stop_training = True
                        model.load_state_dict(previous_model_params)
                        # print('learning <= threshold, stop training.')
                        break
                
                else:
                    break
            
            model.eval()
            valid_loss = []
            valid_accs = []
            
            for batch in val_loader:
                x, y = batch
                
                with torch.no_grad():
                    logits = model(x)
                    
                    if binary:
                        acc = binary_acc(logits, y)
                    else:
                        acc = (logits.argmax(dim=-1) == y).float().mean()
                        
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
            
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # print_info(epoch, epochs, train_loss, train_acc, valid_loss, valid_acc)
                
            if epoch+1 == epochs:
                # print(f'Already trained {epochs} epochs, acceptable')
                return model
            
            if stop_training:
                return model

    except UnboundLocalError:
        # print('Your eta_threshold is setting higher than your learning rate. Reset it with lower one!')
        return None

  
def multiclass_weight_tuning(train_loader=None,
                             val_loader=None,
                             epochs:int=None,
                             model:torch.nn.Module=None,
                             optimizer:torch.optim=None,
                             criterion=None,
                             loss_threshold:float=None,
                             eta_threshold:float=None,
                             lts:int=None,
                             device=None,
                             binary=True):
    
    previous_train_loss = 10000
    # Use try and except to detect whether the eta_threshold is set too high initially
    try:
        for epoch in itertools.count():
            
            model.train().to(device)
            
            previous_model_params = model.state_dict()
            stop_training = False
            
            while True:
                
                train_loss = []
                train_accs = []
                
                for batch in train_loader:
                    
                    x, y = batch
                    
                    logits = model(x.to(device))
                    
                    if binary:
                        loss = criterion(logits, y.to(device))
                    else:
                        loss = criterion(logits, y.to(device))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if binary:
                        acc = binary_acc(logits, y)
                    else:
                        acc = (logits.argmax(dim=-1) == y).float().mean()
                        
                    train_loss.append(loss.item())    
                    train_accs.append(acc)
                
                if isinstance(lts, int):
                    train_loss.sort()
                    train_loss = train_loss[:lts]
                    
                max_train_loss = max(train_loss)
                train_loss = sum(train_loss) / len(train_loss)
                train_acc = sum(train_accs) / len(train_accs)
                
                if eta_threshold is not None:
                    if train_loss < previous_train_loss:
                        optimizer.param_groups[0]['lr'] *= 1.2
                        previous_train_loss = train_loss
                        break
                    
                    if optimizer.param_groups[0]['lr'] > eta_threshold:
                        optimizer.param_groups[0]['lr'] *= 0.7
                        model.load_state_dict(previous_model_params)
                    
                    else:
                        stop_training = True
                        # print('learning rate < threshold')
                        SLFN = 'Unacceptable'    
                        break
                
                else:
                    break
                
            model.eval()
            valid_loss = []
            valid_accs = []
            
            for batch in val_loader:
                x, y = batch
                
                with torch.no_grad():
                    logits = model(x.to(device))
                    
                    if binary:
                        acc = binary_acc(logits, y)
                    else:
                        acc = (logits.argmax(dim=-1) == y).float().mean()
                        
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
            
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # print_info(epoch, epochs, train_loss, train_acc, valid_loss, valid_acc)
                
            if epoch+1 >= epochs:
                # print(f'Already trained {epochs} epochs, unacceptable')
                SLFN = 'Unacceptable'
                return SLFN, model
            
            if stop_training:
                return SLFN, model
            
            if loss_threshold is not None:
                if max_train_loss < loss_threshold:
                    SLFN = 'Acceptable'
                    return SLFN, model
            
    except UnboundLocalError:
            print('Your eta_threshold is setting higher than your learning rate. Reset it with lower one!')
            return 'Error', None


def reorganize_module(model:TwoLayerNN=None,
                      train_loader=None,
                      val_loader=None,
                      criterion=None,
                      reg_epochs:int=None,
                      reg_optimizer=None,
                      reg_loss=None,
                      reg_eta=None,
                      weight_epochs=None,
                      weight_optimizer=None,
                      weight_loss=None,
                      weight_eta=None,
                      l1_lambda=0.001,
                      l2_lambda=0.001,
                      k=1,
                      p=50,
                      device=None):
    while not k>p:
        
        model = multiclass_regularization(train_loader=train_loader,
                                          val_loader=val_loader,
                                          epochs=reg_epochs,
                                          model=model,
                                          optimizer=reg_optimizer,
                                          criterion=criterion,
                                          l1_lambda=l1_lambda,
                                          l2_lambda=l2_lambda,
                                          loss_threshold=reg_loss,
                                          eta_threshold=reg_eta,
                                          device=device)
        
        saved_model = copy.deepcopy(model)
        
        if model.layer_out.weight.data.numel() > 1:
            prune_index = model.layer_1.weight.sum(1).argmin()
            model.del_neuron(index=prune_index)
        
        situation, model = multiclass_weight_tuning(train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    epochs=weight_epochs,
                                                    model=model,
                                                    optimizer=weight_optimizer,
                                                    criterion=criterion,
                                                    loss_threshold=weight_loss,
                                                    eta_threshold=weight_eta,
                                                    device=device)
        
        if situation == 'Unacceptable':
            model = saved_model
            k +=1
            
        elif situation == 'Acceptable':
            print('prune')
            p-= 1
            
    return model


def LTS_module(train_loader=None, model=None, criterion=None, n=None, binary=True):
    
    model.eval()
    valid_loss = []

    for i, batch in enumerate(train_loader.dataset):
        x, y = batch[0].unsqueeze(0), batch[1]
        
        with torch.no_grad():
            logits = model(x)
            
            if binary:
                loss = criterion(logits, y.unsqueeze(0))
            else:
                loss = criterion(logits, y)
                
            valid_loss.append((i, loss.item()))

    picked_loss = []
    # obtaining_LTS
    if isinstance(n, float):
        for index, item in valid_loss:
            if item < n:
                picked_loss.append((index, item))
                
    # selecting_LTS
    if isinstance(n, int):
        valid_loss.sort(key=lambda x: x[1])
        picked_loss = valid_loss[:n]

    picked_index = [i for i, _ in picked_loss]
    n_data_loader = LTS_dataloader(train_loader.dataset, picked_index, train_loader.batch_size)

    return n_data_loader, len(picked_index)


def find_cram_index(model, data_loader, criterion):
    '''
    This function will return the index of wrong prediction data with biggest loss
    '''
    
    model.eval()
    mem = []
    with torch.no_grad():
        for i, data in enumerate(data_loader.dataset):
            x, y = data[0].unsqueeze(0), data[1].squeeze(0)
            y_pred = torch.round(torch.sigmoid(model(x))).squeeze()
            loss = criterion(y, y_pred)
            
            if y_pred != y:
                mem.append((i, loss.item()))
                
    return max(mem, key=lambda mem: mem[1])[0]


def evaluate(train_loader=None, model=None, criterion=None, binary=True):
    
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in train_loader:
        x, y = batch
        
        with torch.no_grad():
            
            logits = model(x)
            
            if binary:
                loss = criterion(logits, y)
                acc = binary_acc(logits, y)
            else:
                loss = criterion(logits, y)
                acc = (logits.argmax(dim=-1) == y).float().mean()
                
            valid_loss.append((loss.item()))
            valid_accs.append(acc)
    
    # print(f'loss:{sum(valid_loss) / len(valid_loss)} | acc:{sum(valid_accs) / len(valid_accs)}')
    return (sum(valid_loss) / len(valid_loss)), (sum(valid_accs) / len(valid_accs))