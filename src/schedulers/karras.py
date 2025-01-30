import torch
from src.nn.utils import rescale_vector

class KarrasScheduler:
    def __init__(self, 
                 num_steps=1, 
                 sigma_min=0.002, 
                 sigma_max=80.0, 
                 sigma_data=0.5, 
                 rho=7.0):
        
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        rho_inv = 1/rho

        print(f'SIGMA_MIN: {self.sigma_min}, SIGMA_MAX: {self.sigma_max}, SIGMA_DATA: {self.sigma_data}')
        steps = torch.arange(num_steps)/max(num_steps-1, 1)
        self.sigmas = (self.sigma_min**rho_inv + 
                       steps*(self.sigma_max**rho_inv - self.sigma_min**rho_inv))**rho
        
    def _check_device(self, tensor):
        if tensor.device != self.sigmas.device:
            self.sigmas = self.sigmas.to(tensor.device)
        
    def get_n_steps(self):
        return self.num_steps
    
    def get_sigmas(self, t=None):
        if t is not None:
            self._check_device(t)
        sigmas = self.sigmas if t is None else self.sigmas[t]
        return sigmas
        
    def get_scalings(self, sigmas):
        c_skip = (self.sigma_data**2) / ((sigmas - self.sigma_min)**2 + self.sigma_data**2)
        c_out = (self.sigma_data * (sigmas - self.sigma_min)) / (self.sigma_data**2 + sigmas**2)**0.5
        c_in = 1/(sigmas**2+self.sigma_data**2)**0.5

        return c_skip, c_out, c_in
    
    def add_noise(self, x, sigmas, noise):
        self._check_device(x)
        x_noisy = x + noise * rescale_vector(sigmas, x)
        return x_noisy
    
    def prepare_starting_data(self, batch, sigmas, noise):
        y = batch.transform['y'](batch.y)
        og_mask = batch.og_mask
        mask = batch.mask

        # create tensor of real values if we have them.
        zeros = torch.zeros_like(y)
        y_real_0 = torch.where(og_mask, y, zeros)

        y_noisy_t = self.add_noise(y_real_0, sigmas, noise)

        # create x_noisy masking out values that we know
        x_noisy_t = torch.where(mask.bool(), zeros, y_noisy_t)
        return x_noisy_t
        
