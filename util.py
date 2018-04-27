import numpy as np
import torch
import torch.nn as nn

def save_model(model,acc):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state,'model_state.pkl')
    print('*** DONE! ***')

