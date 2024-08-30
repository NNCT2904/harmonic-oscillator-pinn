import torch
import matplotlib as plt

def backend_check():
    print('Backend check')

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print('Backend detected: mps')
        device = torch.device('mps')
    else:
        print('Bo backend, use CPU')
        device = torch.device('cpu')

    print('Use .to(device) to send vectors to backend!')

    return device