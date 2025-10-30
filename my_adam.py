import torch

class My_adam(object):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08):
        self._params = list(params)
        device  = self._params[0].device
        # Training hyperparameters
        self._lr = torch.tensor(lr, requires_grad=False).to(device)
        self._betas = torch.tensor(betas, requires_grad=False).to(device)
        self._eps = torch.tensor(eps, requires_grad=False).to(device)
        # Internal state
        self._step = 0
        self._m = list()
        self._v = list()
        for p in self._params:
            self._m.append(torch.zeros_like(p, requires_grad=False).to(device))
            self._v.append(torch.zeros_like(p, requires_grad=False).to(device))

    def step(self):
        self._step += 1
        t = self._step
        beta1, beta2 = self._betas
        for i,p in enumerate(self._params):
            if p.grad is None:
                continue
            g_p = p.grad

            # Update biased first and second moment estimates
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g_p
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * torch.square(g_p)

            # Compute bias-correcting step size
            step_size = self._lr * torch.sqrt(1 - torch.pow(beta2,t)) / (1 - torch.pow(beta1,t))

            # Update parameters
            with torch.no_grad():
                p.sub_(step_size * self._m[i] / (self._v[i].sqrt() + self._eps))

    # Reset gradients to zero
    def zero_grad(self):
        for p in self._params:
            if p.grad is None:
                continue
            p.grad.zero_()


# Infinity norm version of Adam
class My_adamax(My_adam):
    def step(self):
        self._step += 1
        t = self._step
        beta1, beta2 = self._betas
        for i,p in enumerate(self._params):
            if p.grad is None:
                continue
            g_p = p.grad
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g_p

            # Use infinity norm for the second moment, does not require bias correction
            self._v[i] = torch.maximum(beta2 * self._v[i], torch.abs(g_p))
            with torch.no_grad():
                p.sub_(self._lr * self._m[i] / ((self._v[i] + self._eps) * (1 - torch.pow(beta1,t))))


# Adam without bias correction for comparison purposes
class My_adam_no_bias_correction(My_adam):
    def step(self):
        self._step += 1
        t = self._step
        beta1, beta2 = self._betas
        for i,p in enumerate(self._params):
            if p.grad is None:
                continue
            g_p = p.grad
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g_p
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * torch.square(g_p)
            with torch.no_grad():
                p.sub_(self._lr * self._m[i] / (self._v[i].sqrt() + self._eps))