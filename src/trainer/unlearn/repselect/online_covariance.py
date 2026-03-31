import torch

# todo: we can delete this file now


class OnlineCovariance:
    """Minimal Welford-style online mean + covariance tracker (stores upper triangle only)."""

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        self._count = 0
        self.mean = None
        self.half_cov = None

    def add_all(self, xs: torch.Tensor):
        xs = xs.to(dtype=self.dtype)
        n = xs.shape[0]

        if self.mean is None:
            D = xs.shape[1]
            self.mean = torch.zeros(D, dtype=self.dtype, device=xs.device)
            self.half_cov = torch.zeros(D * (D + 1) // 2, dtype=self.dtype, device=xs.device)

        self._count += n
        delta = xs - self.mean
        self.mean.add_(delta.sum(0) / self._count)
        delta2 = xs - self.mean
        full = torch.einsum("ni,nj->ij", delta, delta2) / self._count
        rows, cols = torch.triu_indices(full.shape[0], full.shape[0], device=full.device)
        self.half_cov.mul_((self._count - n) / self._count)
        self.half_cov.add_(full[rows, cols])

    def cov(self) -> torch.Tensor:
        D = self.mean.shape[0]
        rows, cols = torch.triu_indices(D, D, device=self.mean.device)
        full = torch.zeros(D, D, dtype=self.dtype, device=self.mean.device)
        full[rows, cols] = self.half_cov
        full[cols, rows] = self.half_cov
        return full


# class OnlineCovarianceSimple:
#     """Minimal Welford-style online mean + covariance tracker."""

#     def __init__(self, dtype=torch.bfloat16):
#         self.dtype = dtype
#         self._count = 0
#         self.mean = None
#         self.cov = None

#     def add_all(self, xs: torch.Tensor):
#         xs = xs.to(dtype=self.dtype)
#         n = xs.shape[0]

#         if self.mean is None:
#             D = xs.shape[1]
#             self.mean = torch.zeros(D, dtype=self.dtype, device=xs.device)
#             self.cov = torch.zeros(D, D, dtype=self.dtype, device=xs.device)

#         self._count += n
#         delta = xs - self.mean
#         self.mean.add_(delta.sum(0) / self._count)
#         delta2 = xs - self.mean
#         self.cov.mul_((self._count - n) / self._count)
#         self.cov.add_(torch.einsum("ni,nj->ij", delta, delta2) / self._count)



class OnlineCovarianceSimple:
    """Minimal Welford-style online mean + covariance tracker."""
    # also without halving the cov and without mean

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        # self._count = 0
        self.cov = None

    def add_all(self, xs: torch.Tensor):
        xs = xs.to(dtype=self.dtype)
        # n = xs.shape[0]

        if self.cov is None:
            D = xs.shape[1]
            self.cov = torch.zeros(D, D, dtype=self.dtype, device=xs.device)

        # self._count += n
        self.cov += torch.einsum("ni,nj->ij", xs, xs)
