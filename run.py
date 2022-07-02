# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:34:13 2022

@author: njucx
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def rmse(loss):
    loss = math.sqrt(loss)
    return loss

    
batch_size = 256
seq_len = 40
d_model = seq_len

dataset_name = 'FD001'
dataset = get_dataset(dataset_name, seq_len);
test_seq = dataset['lower_test_seq_tensor']
test_label = dataset['lower_test_label_tensor']


model = torch.load('model_FD001_18.pk1').to(device)
#criterion = RMSELoss()
criterion = nn.MSELoss()


def test(model, criterion, batch_size):

    model.eval()  # turn on evaluation mode

    total_test_loss = 0
    pre_result = []  # list(101) -> (50, 128, 1)
    num_batches = test_seq.shape[0] // batch_size
    src_mask = generate_square_subsequent_mask(seq_len).to(device)
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, num_batches*batch_size, batch_size)):
            # compute the loss for the lower-level
            inputs, targets = get_batch(test_seq, test_label, i, batch_size) #[40, 256, 18] [256, 40]
            inputs = inputs.float()
            targets = targets.permute(1, 0)
            targets = torch.unsqueeze(targets, 2).float() # [40, 256, 1]
            predictions = model(inputs, src_mask)   # [40, 256, 1]
            loss = criterion(predictions, targets)                                       
            
            total_test_loss += loss.item()
            pre_result.append(np.array(predictions.cpu()))  # [51, 40, 256, 1]
            
        total_test_loss = total_test_loss / num_batches
        

    return total_test_loss, pre_result


test_loss , pre_result = test(model, criterion, batch_size)
visual(pre_result, dataset_name, seq_len)
singleRUL_visual(pre_result, dataset_name, seq_len, 100)

print("best test loss(MSE):", test_loss)
print("best test loss(RMSE):", rmse(test_loss))

study_store_addr_HI = "sqlite:///%s_fea%s_HI.db" % (dataset_name, str(d_model))
#df = optuna.create_study(study_name='HIpredict_optim_'+ dataset_name, direction="minimize", storage = study_store_addr_HI, load_if_exists=True)  
