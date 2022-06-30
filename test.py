# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import *
from model import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)

plt.style.use('_mpl-gallery')

x = 0.5 + np.arange(100)
test_truth = torch.randn(100)

# plot
fig, ax = plt.subplots()

ax.bar(x, test_truth, width=1, edgecolor="white", linewidth=0.7, label='Truth Data')

ax.set(xlim=(0, 100), xticks=np.arange(1, 100),
       ylim=(0, 100), yticks=np.arange(1, 100))

plt.show()