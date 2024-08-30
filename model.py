import torch.nn as nn
import pennylane as qml

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int):
        super().__init__()
        activation = nn.Tanh

        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class QN(nn.Module):
    '''
    Simple 1 qubit model
    '''

    def __init__(self, N_INPUT: int, N_OUTPUT: int, Q_NODE, N_QUBITS):
        super().__init__()
        self.clayer_1= nn.Linear(N_INPUT, N_QUBITS)
        self.qlayer_1 = Q_NODE
        self.clayer_2 = nn.Linear(N_QUBITS, N_OUTPUT)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer_1(x)
        x = self.clayer_2(x)
        return x