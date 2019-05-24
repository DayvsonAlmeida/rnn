import numpy as np
import sys
#Função de ativação
def sigmoid(x):
    return 1/(1+np.exp(-x))
#Derivada sigmoid
def sigmoid_dx(x):
    return sigmoid(x)*(1-sigmoid(x))

def int2bin(x):
    b = []
    for i in range(3,-1,-1):
        b.append(x//2**i)
        x = x%(2**i)
    return b

def bin2int(x):
    out = 0
    power = len(x)-1
    for i in x:
        out += np.round(i)*2**power
        power -= 1
    return out

class RNN_Elman(object):
    """docstring for RNN."""

    #Construtor
    def __init__(self, epochs=100, lr=0.1, random_state=10):
        np.random.seed(random_state)
        self.layers = None #output de cada layer
        self.nlayers = 0
        self.n_neurons = [] #Qtd neuronios na camada i
        self.weights = None #Matriz de pesos das ligações entre camadas
        self.flag = {'in':False, 'hidden':False, 'out':False}
        self.context = None
        self.input_size = 0
        self.epochs = epochs
        self.lr = lr
        self.errors = []

    #Método para adicionar uma Camada de Entrada
    def addInputLayer(self, n):
        if n<1:
            print('Layer deve possuir pelo menos 1 neurônio!')
        else:
            if self.flag['in'] == True:
                self.n_neurons[0] = n
            else:
                self.n_neurons.append(n)
                self.input_size = n
                self.nlayers += 1
                self.flag['in'] = True

    #Método para adicionar uma Camada Escondida
    def addHiddenLayer(self, n):
        if n<1:
            print('Layer deve possuir pelo menos 1 neurônio!')
        else:
            if self.flag['in'] == True:
                self.n_neurons.append(n)
                self.nlayers += 1
                self.flag['hidden'] = True
            else:
                print('A rede necessita que haja uma camada de entrada!')

    #Método para adicionar uma Camada de Saída
    def addOutputLayer(self, n):
        if n<1:
            print('Layer deve possuir pelo menos 1 neurônio!')
        else:
            if self.flag['hidden'] == True:
                self.n_neurons.append(n)
                self.nlayers += 1
                self.flag['out'] = True
            else:
                print('A rede necessita que haja pelo menos uma camada escondida!')

    #Inicializador de matrizes
    def prepare(self):
        self.layers = [None for i in range(self.nlayers)]
        self.n_neurons[0] += self.n_neurons[1]
        self.weights = [np.random.uniform(low=-1.0, high=1.0, size=(self.n_neurons[i],self.n_neurons[i+1])) for i in range(self.nlayers-1)]
        self.context = np.zeros(self.n_neurons[1])

    #Foward Método
    def _foward(self, input, context):
        #Dados na camada de Entrada
        self.layers[0] = np.concatenate((input, context))
        i=1
        for w in self.weights:
            self.layers[i] = sigmoid(np.dot(self.layers[i-1], w))
            if i==1:
                self.context = self.layers[i]
            i+=1
        return self.layers[i-1]

    #Método para treinamento
    #x: np.array de shape (n_samples, series_size)
    def fit(self, x, y):
        if self.flag['in'] and self.flag['hidden'] and self.flag['out']:
            self.prepare()
            for epoch in range(self.epochs): #Epochs
                total_error = 0
                for xs, ys in zip(x,y): #Pegando o xs e o ys de um indivíduo
                    self.context = np.zeros(self.n_neurons[1]) #Resetando o context para cada indivíduos
                    for pos in range(0, len(xs), self.input_size): #Iterando sobre o xs do indivíduo
                        out = self._foward(xs[pos], self.context)
                        error = ys[pos]-out
                        total_error += (error**2).sum()
                        #Backpropagation slide aula mlp hopfield
                        #Back output to hidden
                        grad_k = [out[k]*(1-out[k])*error[k] for k in range(len(out))]
                        #grad_k = [sigmoid(out[k]*sigmoid_dx(out[k])) for k in range(len(out))]
                        delta_jk = np.zeros((len(self.layers[1]),len(grad_k)))
                        print(self.layers[1].shape)
                        print(self.weights[1].shape)
                        delta_jk = np.dot(self.layers[1], grad_k)
                        '''for k in range(len(grad_k)):
                            for j in range(len(self.layers[1])):
                                delta_jk[j][k] = self.lr*self.layers[1][j]*grad_k[k]
                                self.weights[1][j][k] += delta_jk[j][k]
                        '''
                        #print(delta_jk)
                        sys.exit(0)
                        #Back hidden to input
                        grad_j =[]
                        for j in range(len(self.layers[1])):
                            sum_jk = 0
                            for k in range(len(grad_k)):
                                sum_jk += grad_k[k]*self.weights[1][j][k]
                            grad_j.append(self.layers[1][j]*(1-self.layers[1][j])*sum_jk)

                        delta_ij = np.zeros((len(self.layers[0]),len(grad_j)))
                        for j in range(len(grad_j)):
                            for i in range(len(self.layers[0])):
                                delta_ij[i][j] = self.lr*self.layers[0][i]*grad_j[j]
                                self.weights[0][i][j] += delta_ij[i][j]
                self.errors.append(total_error)
                if(epoch%100==0):
                    print('epoch {} -- error {}'.format(epoch, total_error))
            #np.save('loss.npy', self.errors)
        else:
            print('A rede precisa conter uma camada de entrada, ao menos uma camada escondida e uma camada de saída!')
