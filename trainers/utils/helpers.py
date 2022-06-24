import os
import torch
import importlib
import omegaconf
import core.utils as utils
#   Adapted from: https://github.com/AntheaL/bilevel_opt_code/blob/main/bilevel_opt/helpers.py



def config_to_instance(config_module_name="name",**config):
    module_name = config.pop(config_module_name)
    attr = import_module(module_name)
    if config:
        attr = attr(**config)
    return attr


def import_module(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except:
        try:
            module = import_module(module)
            return getattr(module, attr[1:])
        except:
            return eval(module+attr[1:])

def init_model(model, model_path,dtype, device, is_lower=True):
    if model_path:
        #state_dict_model = torch.load(model_path, map_location='cpu')
        #model = model.load_state_dict(state_dict_model).to('cpu')
        model = torch.load(model_path)

    model = utils.to_type(model, dtype)
    model = model.to(device)

    return model

def save_model(model,model_path):
    torch.save(model.state_dict(), model_path)


def assign_device(device):
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device


def get_dtype(dtype):
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')





