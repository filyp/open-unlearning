import torch as pt
from welford_torch import OnlineCovariance


# todo if using only gate and up proj, we could use just one distr per MLP
# but for simplicity, we can keep it separate for now


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


class TopPCsCollapser:
    def __init__(self, num_proj: int = 10, niter: int = 16):
        self.num_proj = num_proj
        self.niter = niter
        self._reset_vecs()

    def _reset_vecs(self):
        # Reset grads list for next epoch
        self.cache = []

    def add_vecs(self, vecs):
        # self.cache.append(vecs.cpu())  # if VRAM not enough, move to RAM
        self.cache.append(vecs)

    def process_saved_vecs(self):
        # Compute PCA projections for gradients (to collapse)
        pt.cuda.empty_cache()
        if not self.cache:
            return
        vectors_flattened = pt.cat(self.cache)
        self.to_collapse = _get_projections(
            vectors_flattened, self.num_proj, self.niter
        )
        self._reset_vecs()

    def collapse(self, vecs):
        for comp in self.to_collapse:
            vecs = vecs - project_out(vecs, comp)
        return vecs


########################################################


def _get_mahal_dirs(centered, eig_val, eig_vec):
    # Compute Mahalanobis directions using eigendecomposition
    projected = centered @ eig_vec  # (N, D)
    proj_diff = projected - projected / (eig_val / eig_val.min())
    # neg = projected / (eig_val / eig_val.min())
    # neg[:, -24:] = 0
    # proj_diff = projected - neg
    return centered - proj_diff @ eig_vec.T


def _proj_to_mahal_dirs(centered, mahal_dirs):
    mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strenghts * mahal_dirs_norm


class ApproxMahalanobisCollapser:
    def __init__(self, num_proj: int = 1000):
        self.num_proj = num_proj
        self._reset_vecs()

    def _reset_vecs(self):
        # Reset grads list for next epoch
        self.cache = []

    def add_vecs(self, vecs):
        # self.cache.append(vecs.cpu())  # if VRAM not enough, move to RAM
        self.cache.append(vecs)

    def process_saved_vecs(self):
        # Compute PCA projections for gradients (to collapse)
        pt.cuda.empty_cache()
        if not self.cache:
            return
        vectors_flattened = pt.cat(self.cache)

        vectors_flattened = vectors_flattened.to("cuda").float()
        self.mean = vectors_flattened.mean(axis=0)

        centered_vectors = vectors_flattened - self.mean
        _, S, V = pt.pca_lowrank(centered_vectors, self.num_proj)
        self.eig_val = S
        self.pca_components = V

        self._reset_vecs()

    def collapse(self, vecs):
        centered = vecs - self.mean
        mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.pca_components)
        return _proj_to_mahal_dirs(centered, mahal_dirs)


class MahalanobisCollapser:
    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, PCs_to_use: int, device: str):
        self.PCs_to_use = PCs_to_use
        self.device = device
        self._reset_vecs()

    def _reset_vecs(self):
        # Reset online covariance for next epoch
        self.online_cov = OnlineCovariance(device=self.device, dtype=pt.float32)

    def add_vecs(self, vecs):
        self.online_cov.add_all(vecs)

    def process_saved_vecs(self):
        # Extract distribution stats from online covariance
        if self.online_cov.mean is None:
            return
        self.mean = self.online_cov.mean
        self.eig_val = self.online_cov.eig_val[-self.PCs_to_use :]
        self.eig_vec = self.online_cov.eig_vec[:, -self.PCs_to_use :]
        self._reset_vecs()

    def collapse(self, vecs):
        centered = vecs - self.mean
        mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.eig_vec)
        return _proj_to_mahal_dirs(centered, mahal_dirs)

    # def collapse(self, vecs):
    #     centered = vecs - self.mean
    #     projected = centered @ self.eig_vec  # (N, D)

    #     # ! Compute Mahalanobis directions using eigendecomposition
    #     # eig_val goes from smallest to largest
    #     _reg = self.eig_val[-self.PCs_to_use]
    #     eig_val_clamped = self.eig_val.clamp(min=_reg)
    #     mahal_dirs = (projected / eig_val_clamped) @ self.eig_vec.T

    #     # project to mahalanobis directions
    #     mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    #     proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    #     return proj_strenghts * mahal_dirs_norm

    # def collapse(self, vecs):
    #     # we rescale the components directly, rather than projecting to mahalanobis directions
    #     centered = vecs - self.mean
    #     projected = centered @ self.eig_vec  # (N, D)
    #     _reg = self.reg * self.eig_val[-1]
    #     eig_val_clamped = self.eig_val.clamp(min=_reg)
    #     scale = eig_val_clamped / _reg
    #     return (projected / scale) @ self.eig_vec.T

    # mahal_dirs = (projected / (self.eig_val + _reg)) @ self.eig_vec.T  # works similarly good to clamping


# class MahalanobisCollapserInvCov:
#     """Same as MahalanobisCollapser, but uses inverse covariance instead of eigendecomposition.

#     It is minimally faster than using eigencomposition, but it's negligible.

#     Contrary to MahalanobisCollapser, here we are not able to set the regularization relative to the largest eigenvalue.

#     Also MahalanobisCollapser, can be extended to modify or inspect the PCA components individuually.
#     """

#     mean: pt.Tensor
#     inverse_cov: pt.Tensor

#     def __init__(self, reg: float = 3e-3):
#         self.reg = reg
#         self._reset_vecs()

#     def _reset_vecs(self):
#         # Reset online covariance for next epoch
#         self.online_cov = OnlineCovariance(device="cuda")

#     def add_vecs(self, vecs):
#         self.online_cov.add_all(vecs)

#     def process_saved_vecs(self):
#         # Extract distribution stats from online covariance
#         if self.online_cov.mean is None:
#             return
#         self.mean = self.online_cov.mean
#         cov = self.online_cov.cov
#         # Add regularization and compute inverse covariance
#         cov_reg = cov + self.reg * pt.eye(cov.shape[0], device=cov.device)
#         self.inverse_cov = pt.linalg.inv(cov_reg)
#         self._reset_vecs()

#     def collapse(self, vecs):
#         centered = vecs - self.mean
#         # Compute Mahalanobis directions using inverse covariance
#         mahal_dirs = centered @ self.inverse_cov.T

#         # project to mahalanobis directions
#         mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
#         proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
#         return proj_strenghts * mahal_dirs_norm
