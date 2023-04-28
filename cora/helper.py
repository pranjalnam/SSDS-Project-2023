import io
from typing import Union

import torch
from torch.utils.data.distributed import DistributedSampler


def accuracy(model, ds):
    ldr = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    n_correct = 0
    for data in ldr:
        (pixels, labels) = data
        with torch.no_grad():
            outputs = model(pixels)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()
    acc = (n_correct * 1.0) / len(ds)
    return acc

def switch_bw_tensor_and_byte(tensor_or_Byte: Union[torch.Tensor, bytes], tensor_to_Byte = True) -> Union[torch.Tensor, bytes]:
    # assert (tensor_to_Byte and tensor_or_Byte == torch.Tensor) or \
    # (not tensor_to_Byte and isinstance(tensor_or_Byte, bytes))
    if tensor_to_Byte:
        buff = io.BytesIO()
        torch.save(tensor_or_Byte, buff)
        buff.seek(0)
        return buff.read()
    else:
        return torch.load(io.BytesIO(tensor_or_Byte))

def set_model_params_from_flattened_tensor(model, flattened_tensor):
    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.data = flattened_tensor[running_idx: running_idx + param.data.numel()].reshape(param.data.shape)
        running_idx += param.data.numel()


def get_flattened_parameters(model, get_flattened_shape = False):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.data.flatten()
        else:
            total_params = torch.cat((total_params, param.data.flatten()))

    if get_flattened_shape:
        return total_params, total_params.shape
    else:
        return total_params


def get_flattened_params(model, get_flattened_shape=False):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.data.flatten()
        else:
            total_params = torch.cat((total_params, param.data.flatten()))

    if get_flattened_shape:
        return total_params, total_params.shape
    else:
        return total_params


def update_model_params_from_flattened_tensor(model, flattened_tensor):
    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.data = flattened_tensor[running_idx: running_idx + param.data.numel()].reshape(param.data.shape)
        running_idx += param.data.numel()