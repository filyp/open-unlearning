import torch as pt


_triu_indices_cache: dict[tuple, tuple] = {}


def _get_triu_indices(D: int, device: pt.device):
    # saves time by caching the upper triangle indices, once per D and device
    key = (D, device)
    if key not in _triu_indices_cache:
        _triu_indices_cache[key] = pt.triu_indices(D, D, device=device)
    return _triu_indices_cache[key]


class OnlineCovariance:
    """Minimal online mean + covariance tracker (stores upper triangle only).

    Implements the Welford algorithm, but it stores covariance not divided by n.
    Scale doesn't affect any of our computations, so it's fine.
    It requires less operations and doesn't harm numerical stability at all.
    """

    def __init__(self, dtype=pt.bfloat16):
        self.dtype = dtype
        self._count = 0
        self.mean = None
        self.half_cov = None

    def add_all(self, xs: pt.Tensor):
        xs = xs.to(dtype=self.dtype)
        n = xs.shape[0]

        if self.mean is None:
            D = xs.shape[1]
            self.mean = pt.zeros(D, dtype=self.dtype, device=xs.device)
            self.half_cov = pt.zeros(
                D * (D + 1) // 2, dtype=self.dtype, device=xs.device
            )

        self._count += n
        delta = xs - self.mean
        self.mean.add_(delta.sum(0) / self._count)
        delta2 = xs - self.mean
        full = pt.einsum("ni,nj->ij", delta, delta2)
        rows, cols = _get_triu_indices(full.shape[0], full.device)
        self.half_cov.add_(full[rows, cols])

    def cov(self) -> pt.Tensor:
        D = self.mean.shape[0]
        rows, cols = _get_triu_indices(D, self.mean.device)
        full = pt.zeros(D, D, dtype=self.dtype, device=self.mean.device)
        full[rows, cols] = self.half_cov
        full[cols, rows] = self.half_cov
        return full


class BatchedOnlineCovariance:
    """E independent OnlineCovariance trackers stored as batched (E, ...) tensors.
    
    Cov storage is in bfloat16 to save memory, but mean and computation are in float32.
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        # self.dtype = dtype
        self.D = None
        self._count = pt.zeros(num_experts, dtype=pt.long)
        self.mean: pt.Tensor | None = None  # (E, D)
        self.half_cov: pt.Tensor | None = None  # (E, D*(D+1)//2)

    def add_all(self, vecs: pt.Tensor, offsets: pt.Tensor, num_experts: int):
        """vecs: (S, D) sorted by expert; offsets: (E,) cumulative end indices."""
        ends = offsets.tolist()
        starts = [0] + ends[:-1]

        if self.mean is None:
            self.D = vecs.shape[1]
            self.device = vecs.device
            self.mean = pt.zeros(
                num_experts, self.D, dtype=pt.float32, device=self.device
            )
            self.half_cov = pt.zeros(
                num_experts,
                self.D * (self.D + 1) // 2,
                dtype=pt.bfloat16,
                device=self.device,
            )
            self._count = self._count.to(self.device)

        ns = [end - s for s, end in zip(starts, ends)]
        self._count += pt.tensor(ns, dtype=self._count.dtype, device=self.device)

        expert_ids = pt.bucketize(pt.arange(vecs.shape[0], device=self.device), offsets)
        
        vecs = vecs.float()

        for e in range(num_experts):
            s, end = starts[e], ends[e]
            if s == end:
                continue
            xs = vecs[s:end]
            delta = xs - self.mean[e]
            self.mean[e] += delta.sum(0) / self._count[e]
            delta2 = xs - self.mean[e]
            full = pt.einsum("ni,nj->ij", delta, delta2)
            rows, cols = _get_triu_indices(full.shape[0], full.device)
            self.half_cov[e] += full.bfloat16()[rows, cols]

    def cov(self, e: int) -> pt.Tensor:
        rows, cols = _get_triu_indices(self.D, self.device)
        full = pt.zeros(self.D, self.D, dtype=pt.bfloat16, device=self.device)
        full[rows, cols] = self.half_cov[e]
        full[cols, rows] = self.half_cov[e]
        return full


# # this one is similar to Welford algorithm, but it's a bit simler and has less operations, and in practice works just as well
# class OnlineCovariance:
#     """Minimal Welford-style online mean + covariance tracker (stores upper triangle only).

#     To be precise, it stores covariance not divided by n,
#     but scale doesn't affect any of our computations, so it's fine.
#     """

#     def __init__(self, dtype=torch.bfloat16):
#         self.dtype = dtype
#         self._count = 0
#         self.sum = None
#         self.half_cov = None

#     def add_all(self, xs: torch.Tensor):
#         xs = xs.to(dtype=self.dtype)
#         n = xs.shape[0]

#         if self.half_cov is None:
#             D = xs.shape[1]
#             self.sum = torch.zeros(D, dtype=self.dtype, device=xs.device)
#             self.half_cov = torch.zeros(D * (D + 1) // 2, dtype=self.dtype, device=xs.device)

#         self._count += n
#         self.sum += xs.sum(0)
#         delta = xs - self.mean()
#         full = torch.einsum("ni,nj->ij", delta, delta)
#         rows, cols = _get_triu_indices(full.shape[0], full.device)
#         self.half_cov.add_(full[rows, cols])

#     def mean(self) -> torch.Tensor:
#         return self.sum / self._count

#     def cov(self) -> torch.Tensor:
#         D = self.sum.shape[0]
#         rows, cols = _get_triu_indices(D, self.sum.device)
#         full = torch.zeros(D, D, dtype=self.dtype, device=self.sum.device)
#         full[rows, cols] = self.half_cov
#         full[cols, rows] = self.half_cov
#         return full


# # this one computes the proper Welford algorithm
# class OnlineCovariance:
#     """Minimal Welford-style online mean + covariance tracker (stores upper triangle only)."""

#     def __init__(self, dtype=torch.bfloat16):
#         self.dtype = dtype
#         self._count = 0
#         self._mean = None
#         self.half_cov = None

#     def add_all(self, xs: torch.Tensor):
#         xs = xs.to(dtype=self.dtype)
#         n = xs.shape[0]

#         if self._mean is None:
#             D = xs.shape[1]
#             self._mean = torch.zeros(D, dtype=self.dtype, device=xs.device)
#             self.half_cov = torch.zeros(D * (D + 1) // 2, dtype=self.dtype, device=xs.device)

#         self._count += n
#         delta = xs - self._mean
#         self._mean.add_(delta.sum(0) / self._count)
#         delta2 = xs - self._mean
#         full = torch.einsum("ni,nj->ij", delta, delta2) / self._count
#         rows, cols = _get_triu_indices(full.shape[0], full.device)
#         self.half_cov.mul_((self._count - n) / self._count)
#         self.half_cov.add_(full[rows, cols])

#     def mean(self) -> torch.Tensor:
#         return self._mean

#     def cov(self) -> torch.Tensor:
#         D = self._mean.shape[0]
#         rows, cols = _get_triu_indices(D, self._mean.device)
#         full = torch.zeros(D, D, dtype=self.dtype, device=self._mean.device)
#         full[rows, cols] = self.half_cov
#         full[cols, rows] = self.half_cov
#         return full


# class OnlineCovariance:
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


# class OnlineCovarianceSimple:
#     """Minimal Welford-style online mean + covariance tracker."""
#     # also without halving the cov and without mean

#     def __init__(self, dtype=torch.bfloat16):
#         self.dtype = dtype
#         # self._count = 0
#         self.cov = None

#     def add_all(self, xs: torch.Tensor):
#         xs = xs.to(dtype=self.dtype)
#         # n = xs.shape[0]

#         if self.cov is None:
#             D = xs.shape[1]
#             self.cov = torch.zeros(D, D, dtype=self.dtype, device=xs.device)

#         # self._count += n
#         self.cov += torch.einsum("ni,nj->ij", xs, xs)
