from rnn import RNN_Elman, int2bin, bin2int
from numpy import array
import numpy as np
import pandas as pd

input_size, hidden_size = 4, 10
rede = RNN_Elman(epochs=1, lr=0.1, random_state=42)
rede.addInputLayer(input_size)
rede.addHiddenLayer(hidden_size)
rede.addOutputLayer(input_size)

'''
seq_length = 20
data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
x = [data[:-1]]
y = [data[1:]]
'''

x = [[int2bin(i) for i in range(10)]]
y = [[int2bin(i) for i in range(1,11)]]

#print('Arquitetura: num neuronios hidden {}'.format(hidden_size))
rede.fit(x,y)
elem = x[0][0]
serie = []
out = rede._foward(elem, np.zeros(hidden_size))

serie.append(bin2int(out))
print('\n\nPredict\n')
for i in range(20):
    out = rede._foward(elem, rede.context)
    serie.append(bin2int(out))

for i in range(len(y[0])):
    print(bin2int(y[0][i]), serie[i])
