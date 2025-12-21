def _save_act_hook(module, args):
    module.last_act_full = args[0].detach().clone()


def _save_grad_hook(module, args):
    module.last_grad_full = args[0].detach().clone()


def install_hooks(model):
    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            module.register_forward_pre_hook(_save_act_hook)
            module.register_full_backward_pre_hook(_save_grad_hook)


# * it is still useful if we want to do the more efficient calculation of just the top N PCs, not full covariance matrix inversion
# def project_out(base, unwanted):
#     # check dimensions
#     _pos, _stream = base.shape
#     (_stream2,) = unwanted.shape
#     assert _stream == _stream2

#     unwanted = unwanted / unwanted.norm()
#     magnitudes = (base * unwanted).sum(axis=-1)
#     return pt.einsum("t,s->ts", magnitudes, unwanted)


# def _get_projections(vectors_flattened, num_proj=11, niter=16):
#     num_pc = num_proj - 1
#     vectors_flattened = vectors_flattened.to("cuda").float()

#     mean = vectors_flattened.mean(axis=0)

#     if num_proj == 0:
#         return pt.tensor([])
#     elif num_proj == 1:
#         return mean.reshape(1, -1)

#     centered_vectors = vectors_flattened - mean
#     _, S, V = pt.pca_lowrank(centered_vectors, num_pc, niter=niter)
#     pca_components = V.T

#     # return one tensor of mean and the pca components
#     return pt.cat([mean.reshape(1, -1), pca_components], dim=0)


# def get_projections(vector_lists: dict[str, list[pt.Tensor]], num_proj=11, niter=16):
#     # vectors can be either acts or grads
#     to_collapse = {}
#     for n in list(vector_lists.keys()):
#         pt.cuda.empty_cache()
#         cached_vectors = vector_lists.pop(n)
#         if not cached_vectors:
#             continue
#         vectors_flattened = pt.cat(cached_vectors)
#         to_collapse[n] = _get_projections(vectors_flattened, num_proj, niter)

#     return to_collapse


# def project_to_mahalanobis(acts, pca_data):
#     """
#     Project activations onto their Mahalanobis direction.

#     The Mahalanobis direction for each activation is the direction from the mean
#     to that activation in the whitened (PCA) space, transformed back to the original space.
#     This keeps only the component that is maximally dissimilar from the distribution.

#     Args:
#         acts: [batch, dim] tensor of activations
#         pca_data: [num_proj, dim] tensor where row 0 is mean, rows 1: are PCA components

#     Returns:
#         [batch, dim] tensor with only the Mahalanobis direction component
#     """
#     orig_dtype = acts.dtype
#     acts = acts.float()

#     mean = pca_data[0]  # [dim]
#     components = pca_data[1:]  # [num_pc, dim]
#     num_pc = components.shape[0]

#     if num_pc == 0:
#         # No PCA components, just return direction from mean
#         centered = acts - mean
#         norms = centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         directions = centered / norms
#         magnitudes = (acts * directions).sum(dim=-1, keepdim=True)
#         return (magnitudes * directions).to(orig_dtype)

#     # Center the activations
#     centered = acts - mean  # [batch, dim]

#     # Project onto PCA components to get coordinates in PCA space
#     # coords[i, j] = projection of centered[i] onto component[j]
#     coords = pt.einsum("bd,pd->bp", centered, components)  # [batch, num_pc]

#     # The Mahalanobis direction in original space is obtained by:
#     # 1. The direction in whitened space is just the normalized coordinate vector
#     # 2. Transform back: sum of (coord_j / norm) * component_j
#     coord_norms = coords.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
#     normalized_coords = coords / coord_norms  # [batch, num_pc]

#     # Mahalanobis direction in original space (for each sample)
#     mahal_directions = pt.einsum("bp,pd->bd", normalized_coords, components)  # [batch, dim]

#     # Keep only the component along the Mahalanobis direction
#     magnitudes = (acts * mahal_directions).sum(dim=-1, keepdim=True)  # [batch, 1]

#     return (magnitudes * mahal_directions).to(orig_dtype)  # [batch, dim]
