from rnn import RNN_Elman
from numpy import array
import numpy as np
import pandas as pd

input_size, hidden_size = 1, 10
for hidden_size in [15]:
    rede = RNN_Elman(epochs=10000, lr=0.1, random_state=42)
    rede.addInputLayer(input_size)
    rede.addHiddenLayer(hidden_size)
    rede.addOutputLayer(1)

    seq_length = 20
    data_time_steps = np.linspace(2, 10, seq_length + 1)
    data = np.sin(data_time_steps)

    x = [data[:-1]]
    y = [data[1:]]
    print('Arquitetura: num neuronios hidden {}'.format(hidden_size))
    rede.fit(x,y)
    #np.save('loss_i'+str(input_size)+'_h'+str(hidden_size)+'.npy', rede.errors)
