import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os

from src.DataPreprocess import create_dataloader
from src.InitializingModule import InitModel
from src.TrainingAlgorithm import LTS_module, multiclass_weight_tuning, reorganize_module, find_cram_index, evaluate
from src.utils import BinaryFocalLossWithLogits


# parameters
train_name = 'min_cram_max_prune'
datapath = 'data\SPECT_data.txt'
num_data = 20
hidden_size = 5
# criterion = nn.BCEWithLogitsLoss()
criterion = BinaryFocalLossWithLogits(alpha=0.75)
epochs = 10
opt = 'adamw'
learning_rate = 5e-4
eta_threshold = 1e-6
loss_threshold = 0.4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = 10

# initialize logging
formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
logger = logging.getLogger('training experiments')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f'logs/{train_name}.csv')
fh.setFormatter(formatter)
logger.addHandler(fh)

total_cram = []
total_param_bigger_than_n = []
total_weight_tuning = []
total_n_hidden_node = []
val_accuracy = []
train_accuracy = []
val_loss = []
train_loss = []

assert opt in ['adam', 'rmsprop', 'adamw']

for i in range(num_data):
    print('start')
    
    cramming_times = 0
    param_bigger_than_n_times = 0
    weight_tuning_times = 0
    
    # Separate training set and validation set with random_state
    train_loader, val_loader = create_dataloader(datapath, batch_size=32, random_state=random.randint(0, 1440))
    
    N_data = train_loader.dataset.X.shape[0]
    input_size = train_loader.dataset.X.shape[1]
    
    # assume 3% data are outliers
    N_data *= 0.97
    N_data = int(N_data)
    
    # initialize model
    init_model = InitModel(input_size, hidden_size, 1, device)
    model = init_model.init_module_multi_ReLU_WT()
    print('finish model init')
    
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # record training time
    start_time = time.time()
    
    while True:
        _, n_data = LTS_module(train_loader=train_loader, model=model, criterion=criterion, n=0.7)
        
        reg_optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        weight_optimizer = optim.AdamW(model.parameters(), lr=5e-4)
            
        if N_data <= n_data:
            break
        
        param_num = sum(p.numel() for p in model.parameters())
        
        if n_data < param_num:
            # Add reorganizing model (longer)
            model = reorganize_module(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      criterion=criterion,
                                      reg_epochs=epochs*2,
                                      reg_optimizer=reg_optimizer,
                                      reg_loss=0.3,
                                      reg_eta=None,
                                      weight_epochs=epochs*3,
                                      weight_optimizer=weight_optimizer,
                                      weight_loss=0.6,
                                      weight_eta=None,
                                      p=p*3,
                                      device=device)
            
            print(f'big_param, N:{N_data}, n:{n_data}, params:{param_num}')
            param_bigger_than_n_times += 1
            continue
        
        saved_model = copy.deepcopy(model)
        
        situation, model = multiclass_weight_tuning(train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    epochs=epochs,
                                                    model=model,
                                                    optimizer=optimizer,
                                                    criterion=criterion,
                                                    loss_threshold=0.6,
                                                    eta_threshold=None,
                                                    lts=n_data,
                                                    device=device)
        
        if situation == 'Acceptable':
            
            # Add reorganizing model (longer)
            model = reorganize_module(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      criterion=criterion,
                                      reg_epochs=epochs*2,
                                      reg_optimizer=reg_optimizer,
                                      reg_loss=0.3,
                                      reg_eta=None,
                                      weight_epochs=epochs*3,
                                      weight_optimizer=weight_optimizer,
                                      weight_loss=0.6,
                                      weight_eta=None,
                                      p=p*3,
                                      device=device)
            
            param_num = sum(p.numel() for p in model.parameters())
            print(f'weight tuning, N:{N_data}, n:{n_data}, params:{param_num}')
            weight_tuning_times += 1
            continue
        
        model = saved_model
        
        # cramming
        cram_index = find_cram_index(model, train_loader, criterion)
        model.add_neuron(train_loader, cram_index)
        print('cramming')
        cramming_times += 1
        
        # Add reorganizing model (shorter)
        model = reorganize_module(model=model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  criterion=criterion,
                                  reg_epochs=epochs*3,
                                  reg_optimizer=reg_optimizer,
                                  reg_loss=0.3,
                                  reg_eta=None,
                                  weight_epochs=epochs*2,
                                  weight_optimizer=weight_optimizer,
                                  weight_loss=0.6,
                                  weight_eta=None,
                                  p=p,
                                  device=device)
        
    train_time = time.time() - start_time
    
    # evaluate performance
    t_loss, t_accs = evaluate(train_loader, model, criterion)
    v_loss, v_accs = evaluate(val_loader, model, criterion)

    # save info and model
    logger.info(f"{train_time:.2f}{cramming_times},{param_bigger_than_n_times},{weight_tuning_times},{model.layer_out.weight.numel()},{t_accs.item():.4f},{t_loss:.4f},{v_accs.item():.4f},{v_loss:.4f}")
    
    if not os.path.isdir(f'saved_models/{train_name}'):
        os.mkdir(f'saved_models/{train_name}')
        
    torch.save(model.state_dict(), f'saved_models/{train_name}/model_{i}.ckpt')