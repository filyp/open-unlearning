import torch as pt


# * it is still useful if we want to do the more efficient calculation of just the top N PCs, not full covariance matrix inversion
def project_out(base, unwanted):
    # check dimensions
    _pos, _stream = base.shape
    (_stream2,) = unwanted.shape
    assert _stream == _stream2

    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


def _get_projections(vectors_flattened, num_proj=11, niter=16):
    num_pc = num_proj - 1
    vectors_flattened = vectors_flattened.to("cuda").float()

    mean = vectors_flattened.mean(axis=0)

    if num_proj == 0:
        return pt.tensor([])
    elif num_proj == 1:
        return mean.reshape(1, -1)

    centered_vectors = vectors_flattened - mean
    _, S, V = pt.pca_lowrank(centered_vectors, num_pc, niter=niter)
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
