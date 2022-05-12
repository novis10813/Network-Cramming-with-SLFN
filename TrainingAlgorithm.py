import torch
import itertools


# Basic training pipeline
def train(train_loader=None, val_loader=None, model=None, epochs=None, device=None, criterion=None, optimizer=None):
    for epoch in range(epochs):
        
        model.train()
        train_loss = []
        train_accs = []
        
        for batch in train_loader:
            
            x, y = batch
            
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (logits.argmax(dim=-1) == y.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        model.eval()

        valid_loss = []
        valid_accs = []
        
        for batch in val_loader:
            imgs, labels = batch
            
            with torch.no_grad():
                logits = model(imgs.to(device))
                
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                
                valid_loss.append(loss.item())
                valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
    
    return model


# Universal training process
def universal_training(train_loader=None,
                       val_loader=None,
                       model=None,
                       epochs=None,
                       criterion=None,
                       optimizer=None,
                       device=None,
                       loss_threshold=0.5,
                       eta_threshold=0.008,
                       l1_lambda=None,
                       l2_lambda=None,
                       model_type='regularizing'):
    '''
    Args:
        train_loader: Pytorch trainloader object.
        val_loader: Pytorch valloader object.
        model: A neural network.
        epochs: If None, it will not stop by the limitation of epoch, otherwise the training process will stop at your desired epoch.
        criterion: Usually MSE loss.
        optimizer: Training optimizer.
        device: cpu or cuda
        loss_threshold: stopping criteria for training loss.
        eta_threshold: stopping criteria for learning rate.
        l1_lambda: if None, it won't apply l1 regularization to the model.
        l2_lambda: if None, it won't apply l2 regularization to the model.
        model_type: decide the model type is 'regularizing' or 'weight_tuning'.
    '''
    
    previous_train_loss = 10000    

    for epoch in itertools.count():
        
        model.train()
        
        previous_model_params = model.state_dict()
        stop_training = False
        
        # The mechanism of identifying the neighborhood of an undesired attractor
        while optimizer.param_groups[0]['lr'] > eta_threshold:
            
            train_loss = []
            train_accs = []
            
            for batch in train_loader:
                
                x, y = batch
                
                logits = model(x.to(device))
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
                
                acc = (logits.argmax(dim=-1) == y.to(device)).float().mean()
                train_loss.append(loss.item())
                train_accs.append(acc)
            
            max_train_loss = max(train_loss)
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            
            if model_type == 'regularizing':
                if train_loss <= previous_train_loss:
                    if max_train_loss < loss_threshold:
                        optimizer.param_groups[0]['lr'] *= 1.2
                        previous_train_loss = train_loss
                        break
                    
                    else:
                        model.load_state_dict(previous_model_params)
                        stop_training = True
                        SLFN = 'Acceptable'
                        break
            
            elif model_type == 'weight_tuning':
                if train_loss <= previous_train_loss:
                    if max_train_loss < loss_threshold:
                        optimizer.param_groups[0]['lr'] *= 1.2
                        previous_train_loss = train_loss
                        break
                    
                    else:
                        model.load_state_dict(previous_model_params)
                        stop_training = True
                        SLFN = 'Acceptable'
                        break
            
            optimizer.param_groups[0]['lr'] *= 0.7
            model.load_state_dict(previous_model_params)
            
        else:
            # eta < threshold
            stop_training = True
            if model_type == 'regularizing':
                model.load_state_dict(previous_model_params)
                SLFN = 'Acceptable'
            
            elif model_type == 'weight_tuning':
                SLFN = 'Unacceptable'
        
        # Use try and except to detect whether the eta_threshold is set too high initially
        try:        
            model.eval()
            valid_loss = []
            valid_accs = []
            
            for batch in val_loader:
                imgs, labels = batch
                
                with torch.no_grad():
                    logits = model(imgs.to(device))
                    
                    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
            
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            if epochs is None:
                print(f'[ {epoch+1} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
            
            else:
                print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
                
                if epoch+1 == epochs:
                    print(f'Already trained {epochs} epochs, acceptable')
                    if model_type == 'regularizing':
                        SLFN = 'Acceptable'
                        return SLFN
                    
                    elif model_type == 'weight_tuning':
                        SLFN = 'Unacceptable'
                        return SLFN
                    
                    else:
                        return 'weight_type_error'
            
        except UnboundLocalError:
            print('Your eta_threshold is setting higher than your learning rate. Reset it with lower one!')
            return 'Error'
        
        # stopping criterion
        if stop_training:
            print('Restore previous model weights, stop training.')
            return SLFN


class TrainingAlgo:
    def __init__(self,
                 train_loader=None,
                 val_loader=None,
                 criterion=None,
                 device=None):
        '''
        Args:
            train_loader: Pytorch trainloader object.
            val_loader: Pytorch valloader object.
            epochs: If None, it will not stop by the limitation of epoch, otherwise the training process will stop at your desired epoch.
            criterion: Usually MSE loss.
        '''
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device=device
    
    def multiclass_regularization(self,
                                epochs:int=None,
                                model:torch.nn.Module=None,
                                optimizer:torch.optim=None,
                                l1_lambda:float=None,
                                l2_lambda:float=None,
                                loss_threshold:float=None,
                                eta_threshold:float=None):
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
        model.to(self.device)
        previous_train_loss = 10000
        print('--------initializing regularization--------')
        try:
            for epoch in itertools.count():
                
                model.train()
                
                previous_model_params = model.state_dict()
                stop_training = False
                
                while True:
                    
                    train_loss = []
                    train_accs = []
                    
                    for batch in self.train_loader:
                        
                        x, y = batch
                        
                        logits = model(x)
                        loss = self.criterion(logits, y.to(torch.long))
                        
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
                        
                        acc = (logits.argmax(dim=-1) == y.to(torch.long)).float().mean()
                        
                        train_loss.append(loss.item())
                        train_accs.append(acc)
                    
                    max_train_loss = max(train_loss)
                    train_loss = sum(train_loss) / len(train_loss)
                    train_acc = sum(train_accs) / len(train_accs)
                    
                    if train_loss <= previous_train_loss:
                        if max_train_loss < loss_threshold:
                            optimizer.param_groups[0]['lr'] *= 1.2
                            previous_train_loss = train_loss
                            break
                        
                        else:
                            model.load_state_dict(previous_model_params)
                            stop_training = True
                            print(f'max loss:{max_train_loss} > threshold{loss_threshold}, stop training.')
                            break
                    
                    if optimizer.param_groups[0]['lr'] > eta_threshold:
                        optimizer.param_groups[0]['lr'] *= 0.8
                        model.load_state_dict(previous_model_params)
                    
                    else:
                        stop_training = True
                        model.load_state_dict(previous_model_params)
                        print('learning <= threshold, stop training.')
                        break
                
                model.eval()
                valid_loss = []
                valid_accs = []
                
                for batch in self.val_loader:
                    x, y = batch
                    
                    with torch.no_grad():
                        logits = model(x)
                        
                        acc = (logits.argmax(dim=-1) == y.to(torch.long)).float().mean()
                        valid_loss.append(loss.item())
                        valid_accs.append(acc)
                
                valid_loss = sum(valid_loss) / len(valid_loss)
                valid_acc = sum(valid_accs) / len(valid_accs)

                if epochs is None:
                    print(f'[ {epoch+1} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
                
                else:
                    print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
                    
                    if epoch+1 == epochs:
                        print(f'Already trained {epochs} epochs, acceptable')
                        return model
                
                if stop_training:
                    return model

        except UnboundLocalError:
            print('Your eta_threshold is setting higher than your learning rate. Reset it with lower one!')
            return None
        
    def multiclass_weight_tuning(self,
                                epochs:int=None,
                                model:torch.nn.Module=None,
                                optimizer:torch.optim=None,
                                loss_threshold:float=None,
                                eta_threshold:float=None):
        
        model.to(self.device)
        previous_train_loss = 10000
        print('--------initializing weight tuning--------')
        # Use try and except to detect whether the eta_threshold is set too high initially
        try:
            for epoch in itertools.count():
                
                model.train()
                
                previous_model_params = model.state_dict()
                stop_training = False
                
                while True:
                    
                    train_loss = []
                    train_accs = []
                    
                    for batch in self.train_loader:
                        
                        x, y = batch
                        
                        logits = model(x)
                        loss = self.criterion(logits, y.to(torch.long))
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        acc = (logits.argmax(dim=-1) == y.to(torch.long)).float().mean()
                        train_loss.append(loss.item())
                        train_accs.append(acc)
                        
                    max_train_loss = max(train_loss)
                    train_loss = sum(train_loss) / len(train_loss)
                    train_acc = sum(train_accs) / len(train_accs)
                    
                    if train_loss < previous_train_loss:
                        optimizer.param_groups[0]['lr'] *= 1.1
                        previous_train_loss = train_loss
                        break
                    
                    if optimizer.param_groups[0]['lr'] > eta_threshold:
                        optimizer.param_groups[0]['lr'] *= 0.7
                        model.load_state_dict(previous_model_params)
                    
                    else:
                        stop_training = True
                        print('learning rate < threshold')
                        SLFN = 'Unacceptable'    
                        break
                      
                model.eval()
                valid_loss = []
                valid_accs = []
                
                for batch in self.val_loader:
                    x, y = batch
                    
                    with torch.no_grad():
                        logits = model(x)
                        
                        acc = (logits.argmax(dim=-1) == y.to(torch.long)).float().mean()
                        valid_loss.append(loss.item())
                        valid_accs.append(acc)
                
                valid_loss = sum(valid_loss) / len(valid_loss)
                valid_acc = sum(valid_accs) / len(valid_accs)

                if epochs is None:
                    print(f'[ {epoch+1} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
                
                else:
                    print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
                    
                    if epoch+1 == epochs:
                        print(f'Already trained {epochs} epochs, unacceptable')
                        SLFN = 'Unacceptable'
                        return SLFN, model
                
                if stop_training:
                    return SLFN, model
                
                if max_train_loss < loss_threshold:
                    SLFN = 'Acceptable'
                    return SLFN, model
                
        except UnboundLocalError:
                print('Your eta_threshold is setting higher than your learning rate. Reset it with lower one!')
                return 'Error', None
    
    def _binary_acc(self, y_pred, y_true):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_result_sum = (y_pred_tag == y_true).sum().float()
        acc = correct_result_sum / y_true.shape[0]
        
        return acc