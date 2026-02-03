import torch as pt


def save_act_input(module, args, output):
    assert isinstance(args, tuple)
    assert len(args) == 1
    if module.training:
        module.last_act_input = args[0]
    else:
        module.last_act_input = None  # clean automatically


def save_act_output(module, args, output):
    assert isinstance(output, pt.Tensor)
    if module.training:
        module.last_act_output = output
    else:
        module.last_act_output = None  # clean automatically


def save_grad_input(module, grad_input, grad_output):
    assert isinstance(grad_input, tuple)
    assert len(grad_input) == 1
    if module.training:
        module.last_grad_input = grad_input[0]
    else:
        module.last_grad_input = None  # clean automatically


def save_grad_output(module, grad_input, grad_output):
    assert isinstance(grad_output, tuple)
    assert len(grad_output) == 1
    if module.training:
        module.last_grad_output = grad_output[0]
    else:
        module.last_grad_output = None  # clean automatically
