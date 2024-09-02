import torch
import math
from ksddescent import ksdd_lbfgs, ksdd_gradient
from ksddescent.contenders import svgd, mmd_lbfgs
import matplotlib.pyplot as plt
import numpy as np

from ksd_descent_gaussian import ksdd_lbfgs_gaussian


def score(x):
    return -x / 0.3

def potential(x):
    return (x ** 2).sum(dim=1)


# seems to be the target
def sampler(n_points):
    return math.sqrt(0.3) * torch.randn(n_points, 2)


n_samples = 50
p = 2 # dimension

# Define initial parameters for the Gaussian distribution
m0 = torch.tensor([0.0, 0.0])  # mean
L0 = torch.tensor([[0.5, 0.0], [0.1, 0.4]])  # lower triangular matrix for covariance

bw = 0.1 # bandwidth of the kernel

# call the Gaussian KSD descent
m_ksd, L_ksd, m_traj, L_traj, _ = ksdd_lbfgs_gaussian(m0.clone(), L0.clone(), n_samples, score, bw=bw, store=True)


# Plots

# Visualize the Gaussian distribution defined by m_ksd and L_ksd

# Generate a grid of points
x_ = np.linspace(-1.2, 1.2, 100)
y_ = np.linspace(-1.2, 1.2, 100)
X, Y = np.meshgrid(x_, y_)
XX = torch.tensor(np.array([X.ravel(), Y.ravel()]).T, dtype=torch.float32)

# Compute the Gaussian density
cov_ksd = torch.matmul(L_ksd, L_ksd.T)
inv_cov_ksd = torch.inverse(cov_ksd)
diff = XX - m_ksd
exponent = -0.5 * torch.sum(diff @ inv_cov_ksd * diff, dim=1)
Z = torch.exp(exponent).reshape(X.shape).detach().numpy()

# Plot the Gaussian density
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(label="Density")
plt.scatter(m_ksd[0], m_ksd[1], color="red", label="Mean (m_ksd)")
plt.title("Gaussian Distribution with KSD-optimized Parameters")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()



# make another plot. this time, we plot the contour of the Gaussian within one standard deviation
# Plot the contour of the Gaussian within one standard deviation
plt.figure(figsize=(6, 6))
# plt.contourf(X, Y, Z, levels=20, cmap="viridis")
# plt.colorbar(label="Density")
plt.contour(X, Y, Z, levels=[np.exp(-0.5)], colors="red", linestyles="dashed", label="1 Std Dev Contour")
plt.scatter(m_ksd[0], m_ksd[1], color="red", label="Mean (m_ksd)")
plt.title("Gaussian Distribution with KSD-optimized Parameters (1 Std Dev Contour)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
