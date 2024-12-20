import torch
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

def backend_check(backend='Auto'):
    print('Backend check')
    print('Use .to(device) to send vectors to backend!')
    if backend=='Auto':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Backend detected: mps')
            device = torch.device('mps')
            return device
        if torch.backends.cuda.is_built():
            print('Backend detected: cuda')
            device = torch.device('cuda')    
            return device    
        else:
            print('No backend, use CPU')
            device = torch.device('cpu')
            return device
    else:
        print(f'Use backend "{backend}"')
        device = torch.device(backend)    
    return device


### Draw a circuit
#   Lots of styles apply, e.g. 'black_white', 'black_white_dark', 'sketch', 
#     'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 
#     'default', we can even use 'rcParams' to redefine all attributes

def draw_circuit(circuit, fontsize=20, style='pennylane', expansion_strategy=None, scale=None, title=None, decimals=2):
    def _draw_circuit(*args, **kwargs):
        nonlocal circuit, fontsize, style, expansion_strategy, scale, title
        qml.drawer.use_style(style)
        if expansion_strategy is None:
            expansion_strategy = circuit.expansion_strategy
        fig, ax = qml.draw_mpl(circuit, decimals=decimals, expansion_strategy=expansion_strategy)(*args, **kwargs)
        if scale is not None:
            dpi = fig.get_dpi()
            fig.set_dpi(dpi*scale)
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        plt.show()
    return _draw_circuit

class WeightClipper(object):
    def __call__(self, module, param_range=[0, np.pi]):
        if hasattr(module, 'weights'):
            w = module.weights.data
            w = w.clamp(param_range[0], param_range[1])
            module.weights.data = w

def custom_weights(m): 
    # Usage:
    # qlayer.apply(custom_weights)
    # qlayer.weights
    torch.nn.init.uniform_(m.weights, 0, np.pi) 