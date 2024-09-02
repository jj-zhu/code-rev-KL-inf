# Author: JZ
import torch
import numpy as np
from ksddescent.kernels import (imq_kernel, gaussian_stein_kernel_single,
                      linear_stein_kernel)

from scipy.optimize import fmin_l_bfgs_b
from time import time

def ksdd_lbfgs_gaussian(mean, cov, n_samples, score, kernel='gaussian', bw=1.,
                        max_iter=10000, tol=1e-12, beta=.5,
                        store=False, verbose=False):
    m = mean.clone().detach().numpy()
    L = cov.clone().detach().numpy()
    p = m.shape[0] # because we are using a Gaussian, which is fully parameterized by its mean and covariance matrix; no need to pass all the samples
    
    if store:
        class callback_store():
            def __init__(self):
                self.t0 = time()
                self.m_mem = []
                self.L_mem = []
                self.timer = []

            def __call__(self, theta):
                m = theta[:p]
                L = theta[p:].reshape(p, p)
                self.m_mem.append(np.copy(m))
                self.L_mem.append(np.copy(L))
                self.timer.append(time() - self.t0)

            def get_output(self):
                m_storage = [torch.tensor(m, dtype=torch.float32) for m in self.m_mem]
                L_storage = [torch.tensor(L, dtype=torch.float32) for L in self.L_mem]
                return m_storage, L_storage, self.timer
        callback = callback_store()
    else:
        callback = None
    
    def loss_and_grad_gaussian(theta_np):
        # Split the input array into m and L
        m_numpy = theta_np[:p]
        L_numpy = theta_np[p:].reshape(p, p)

        m = torch.tensor(m_numpy, dtype=torch.float32, requires_grad=True)
        L = torch.tensor(L_numpy, dtype=torch.float32, requires_grad=True)

        # use reparameterization trick to sample from the std Gaussian
        # TODO: we see this sampling is called every time we call the loss_and_grad_gaussian function, hence it (might) be stochastic
        epsilon = torch.randn(n_samples, p)

        x = m + torch.matmul(L, epsilon.T).T
        scores_x = score(x)

        kernel = 'gaussian'
        if kernel == 'gaussian':
            stein_kernel = gaussian_stein_kernel_single(x, scores_x, bw)
        elif kernel == 'imq':
            stein_kernel = imq_kernel(x, x, scores_x, scores_x, bw, beta=beta)
        else:
            stein_kernel = linear_stein_kernel(x, x, scores_x, scores_x)

        loss = stein_kernel.sum() / n_samples ** 2
        loss.backward()
        grad_m = m.grad
        grad_L = L.grad

        # Flatten and concatenate gradients
        grad = np.concatenate([np.float64(grad_m.numpy().ravel()), 
                               np.float64(grad_L.numpy().ravel())])

        return loss.item(), grad
    
    t0 = time()

    # Flatten and concatenate m and L for input
    theta_init = np.concatenate([m.ravel(), L.ravel()])
    
    # Optimize
    theta_final, f, d = fmin_l_bfgs_b(loss_and_grad_gaussian, theta_init, maxiter=max_iter,
                                      factr=tol, epsilon=1e-12, pgtol=1e-10,
                                      callback=callback)
    
    # Unpack the result
    m_final = theta_final[:p]
    L_final = theta_final[p:].reshape(p, p)

    if verbose:
        print('Took %.2f sec, %d iterations, loss = %.2e' % (time() - t0, d['nit'], f))

    # Convert results to torch tensors
    m_output = torch.tensor(m_final, dtype=torch.float32)
    L_output = torch.tensor(L_final, dtype=torch.float32)

    if store:
        m_storage, L_storage, timer = callback.get_output()
        return m_output, L_output, m_storage, L_storage, timer
    else:
        return m_output, L_output
    
        
    
