# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create elongated gaussian cloud
n_points = 100
cov = np.array([[2.0, 1.5], [1.5, 2.0]])
mean = np.array([0, 0])
points = np.random.multivariate_normal(mean, cov, n_points)

# # Add some outliers
# outliers = np.random.uniform(-4, 4, (50, 2))
# points = np.vstack([points, outliers])

# Compute statistics
centered = points - mean
precision = np.linalg.inv(cov)

# Mahalanobis directions: precision @ centered.T gives direction in Mahal space
mahal_dirs = (precision @ centered.T).T
mahal_dirs_norm = mahal_dirs / (np.linalg.norm(mahal_dirs, axis=1, keepdims=True) + 1e-8)
centered_norm = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)

# %%
# Five distance metrics
metrics = {
    "centered * mahal_dirs_norm\n(BEST)": (centered * mahal_dirs_norm).sum(axis=1),
    "centered_norm * mahal_dirs_norm": (centered_norm * mahal_dirs_norm).sum(axis=1),
    "centered * mahal_dirs\n(~Mahal dist of acts)": (centered * mahal_dirs).sum(axis=1),
    "mahal_dirs_norm @ precision\n(Mahal dist of direction)": ((mahal_dirs_norm @ precision) * mahal_dirs_norm).sum(axis=1),
    "centered_norm * mahal_dirs\n(WORST)": (centered_norm * mahal_dirs).sum(axis=1),
}

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, (name, dists) in zip(axes, metrics.items()):
    # Threshold at 75th percentile for visibility
    threshold = np.percentile(dists, 75)
    selected = dists > threshold

    ax.scatter(points[~selected, 0], points[~selected, 1], c='blue', alpha=0.3, s=10, label='below')
    ax.scatter(points[selected, 0], points[selected, 1], c='red', alpha=0.7, s=20, label='selected')

    # Draw mahalanobis direction for selected points
    for i in np.where(selected)[0]:
        # Project centered onto mahal_dirs_norm
        proj_len = np.dot(centered[i], mahal_dirs_norm[i])
        proj = mahal_dirs_norm[i] * proj_len
        perp = centered[i] - proj  # perpendicular component
        ax.plot([0, proj[0]], [0, proj[1]], c='green', alpha=0.3, lw=0.5)
        ax.plot([proj[0], proj[0] + perp[0]], [proj[1], proj[1] + perp[1]], c='orange', alpha=0.3, lw=0.5)

    # # Draw unnormalized mahalanobis direction and connection to point
    # for i in np.where(selected)[0]:
    #     m_dir = mahal_dirs[i]  # unnormalized
    #     p = centered[i]
    #     ax.plot([0, m_dir[0]], [0, m_dir[1]], c='green', alpha=0.3, lw=0.5)
    #     ax.plot([m_dir[0], p[0]], [m_dir[1], p[1]], c='orange', alpha=0.3, lw=0.5)

    ax.set_title(name, fontsize=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axhline(0, c='gray', lw=0.5)
    ax.axvline(0, c='gray', lw=0.5)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('mahal_visualization.png', dpi=150)
plt.show()

# %%