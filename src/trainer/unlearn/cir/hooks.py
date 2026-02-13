# TODO we can probably remove this file later, and merge it to cir_trainer


# def save_act_input(module, args, output):
#     global use_hooks
#     if not use_hooks:
#         return
#     assert isinstance(args, tuple)
#     assert len(args) == 1
#     module.last_act_input = args[0]


# def save_act_output(module, args, output):
#     global use_hooks
#     if not use_hooks:
#         return
#     assert isinstance(output, pt.Tensor)
#     module.last_act_output = output


# def save_grad_input(module, grad_input, grad_output):
#     global use_hooks
#     if not use_hooks:
#         return
#     assert isinstance(grad_input, tuple)
#     assert len(grad_input) == 1
#     module.last_grad_input = grad_input[0]


# def save_grad_output(module, grad_input, grad_output):
#     global use_hooks
#     if not use_hooks:
#         return
#     assert isinstance(grad_output, tuple)
#     assert len(grad_output) == 1
#     module.last_grad_output = grad_output[0]
