import torch as pt
from transformers import AutoModelForCausalLM

from trainer.unlearn.cir.cir_utils import trainable_modules


def project_out(base, unwanted):
    # check dimensions
    _pos, _stream = base.shape
    (_stream2,) = unwanted.shape
    assert _stream == _stream2

    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


def save_act_hook(module, args):
    module.last_act_full = args[0].detach().clone()


def save_grad_hook(module, args):
    module.last_grad_full = args[0].detach().clone()


def install_hooks(model):
    for n, module in trainable_modules(model):
        module.register_forward_pre_hook(save_act_hook)
        module.register_full_backward_pre_hook(save_grad_hook)


def get_last_act(module, attn_mask, cut_off_tokens=1):
    # ignore BOS token and the last token
    act = module.last_act_full[:, cut_off_tokens:-1]
    final_mask = attn_mask.bool()[:, cut_off_tokens:-1]
    return act[final_mask]


def get_last_grad(module, attn_mask, cut_off_tokens=1):
    # ignore BOS token and the last token
    grad = module.last_grad_full[:, cut_off_tokens:-1]
    final_mask = attn_mask.bool()[:, cut_off_tokens:-1]
    return grad[final_mask]


def _get_projections(vectors_flattened, num_proj=11, niter=16):
    num_pc = num_proj - 1
    vectors_flattened = vectors_flattened.to("cuda").float()

    mean = vectors_flattened.mean(axis=0)
    
    if num_proj == 0:
        return pt.tensor([])
    elif num_proj == 1:
        return mean.reshape(1, -1)

    _, S, V = pt.pca_lowrank(vectors_flattened, num_pc, niter=niter)
    pca_components = V.T

    # return one tensor of mean and the pca components
    return pt.cat([mean.reshape(1, -1), pca_components], dim=0)


def get_projections(vector_lists: dict[str, list[pt.Tensor]], num_proj=11, niter=16):
    # vectors can be either acts or grads
    to_collapse = {}
    for n in list(vector_lists.keys()):
        pt.cuda.empty_cache()
        cached_vectors = vector_lists.pop(n)
        if not cached_vectors:
            continue
        vectors_flattened = pt.cat(cached_vectors)
        to_collapse[n] = _get_projections(vectors_flattened, num_proj, niter)

    return to_collapse

