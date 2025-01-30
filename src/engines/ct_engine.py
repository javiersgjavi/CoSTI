import math
import torch
import numpy as np

from src.engines.engine_template import CustomEngine
from src.nn.unet import UnetContinuous
from src.schedulers.karras import KarrasScheduler
from src.schedulers.curriculum import ExpCurriculumScheduler, BaseCurriculumScheduler, LinearCurriculumScheduler, CurriculumSchedulerPretraining, ConstantScheduler
from src.data.missing_patterns_gen import MissingPatternGenerator

from tsl.metrics import torch as torch_metrics

from copy import deepcopy
from src.nn.utils import rescale_vector

def prepare_y(batch):
    y = batch.transform['y'](batch.y)
    og_mask = batch.og_mask

    zeros = torch.zeros_like(y)
    y = torch.where(og_mask, y, zeros)
    return y

class PseudoHaberLossWeightedMasked:
    def __init__(self, c=None):
        self.c = c

    def weight_function(self, sigmas_0, sigmas_1, mask):
        return 1/(sigmas_1[mask] - sigmas_0[mask])

    def __call__(self, pred, target, mask, sigmas, sigmas_1):
        c = self.c if self.c is not None else 0.00054*torch.sqrt(torch.tensor(torch.numel(pred[0])))
        loss_weights = self.weight_function(sigmas, sigmas_1, mask)

        loss = torch.sqrt((pred[mask] - target[mask])**2 + c**2) - c
        loss = (loss * loss_weights).mean()
        return loss
    
class TimestepGeneratorLognormal:
    def __init__(self, p_mean=-1.1, p_std=2.0, sigmas=None):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigmas = sigmas

        arg1 = torch.erf((torch.log(sigmas[1:]) - p_mean)
                 /(math.sqrt(2)*p_std))

        arg2 = torch.erf((torch.log(sigmas[:-1]) - p_mean)
                 /(math.sqrt(2)*p_std))

        self.pdf = arg1 - arg2
        self.pdf = self.pdf / self.pdf.sum()

    def __call__(self, batch_size):
        return torch.multinomial(self.pdf, num_samples=batch_size, replacement=True)
    
class CT_Engine(CustomEngine):
    def __init__(self, total_iterations, adj, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = kwargs['model_kwargs']
        curriculum_kwargs = cfg['curriculum_kwargs']
        self.tg_kwargs = cfg['timestep_generator_kwargs']
        self.scheduler_kwargs = cfg['scheduler_kwargs']
        dataset_kwargs = cfg['dataset']
        missing_pattern_kwargs = cfg['missing_pattern']
        model_config = cfg['model_config']
        self.total_iterations = total_iterations

        self.mpg = MissingPatternGenerator(
            strategy1=missing_pattern_kwargs['strategy1'],
            strategy2=missing_pattern_kwargs['strategy2'],
            seq_len=dataset_kwargs['time_steps'],
            hist_patterns=cfg['hist_patterns']
        )

        self.model = UnetContinuous(
            n_nodes=dataset_kwargs['num_nodes'],
            t_steps=dataset_kwargs['time_steps'],
            adj=adj,
            **model_config
        )

        # Hay que inicializarlo bien
        self.curriculum_scheduler = LinearCurriculumScheduler(k_max=self.total_iterations, **curriculum_kwargs)
        self.scheduler = KarrasScheduler(num_steps=self.curriculum_scheduler(0), **self.scheduler_kwargs)
        self.t_generator = TimestepGeneratorLognormal(sigmas=self.scheduler.get_sigmas(), **self.tg_kwargs)

        self.test_sigmas = [self.scheduler.sigma_max]
        self.loss_fn = PseudoHaberLossWeightedMasked()
        self.mse_loss = torch_metrics.mse
        self.mae_loss = torch_metrics.mae

    def change_sigmas_predict(self, sigmas):
        self.test_sigmas = sigmas

    def define_scheduler_curriculum(self):
        train_steps = self.curriculum_scheduler(self.global_step)
        if train_steps != self.scheduler.get_n_steps():
            self.scheduler = KarrasScheduler(num_steps=train_steps, **self.scheduler_kwargs)
            self.t_generator = TimestepGeneratorLognormal(sigmas=self.scheduler.get_sigmas(), **self.tg_kwargs)
        
        self.log(
            'curriculum',
            train_steps,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=True
        )

        self.log(
            'k',
            self.global_step,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False
        )

    def forward_cm(self, x_noisy, x_itp, mask, sigmas, sigmas_scaled):
        c_skip, c_out, c_in = self.scheduler.get_scalings(sigmas_scaled)
        c_noise = sigmas.log()/4
        pred = self.model(c_in*x_noisy, x_itp, mask, c_noise)
        pred = c_skip * x_noisy + c_out * pred
        return pred
    
    def _impute_batch(self, batch):
        x_itp = batch['x_interpolated']
        mask = batch['mask']
        x = batch['x']
        x = torch.where(mask, x, torch.zeros_like(x))

        noise = torch.randn_like(x)
        sigmas = (torch.ones(x.shape[0]) * self.test_sigmas[0]).to(x.device)
        sigmas_scaled = rescale_vector(sigmas, x)

        x_noisy = x + noise * sigmas_scaled

        prediction = self.forward_cm(x_noisy, x_itp, mask, sigmas, sigmas_scaled)
        prediction = torch.where(mask, x, prediction)

        for sigma in self.test_sigmas[1:]:
            noise = torch.randn_like(x)
            sigma_noise = (sigma**2 - self.scheduler.sigma_min**2)**0.5

            sigmas_noise = (torch.ones(x.shape[0]) * sigma_noise).to(x.device)
            sigmas = (torch.ones(x.shape[0]) * sigma).to(x.device)
            sigmas_noise_scaled = rescale_vector(sigmas_noise, x)
            sigmas_scaled = rescale_vector(sigmas, x)

            x_noisy = prediction + noise * sigmas_noise_scaled
            prediction = self.forward_cm(x_noisy, x_itp, mask, sigmas, sigmas_scaled)
            prediction = torch.where(mask, x, prediction)

        return prediction
    
    def generate_median_imputation(self, batch, n=100):
        predictions = []
        for _ in range(n):
            pred = self._impute_batch(batch)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=-1)
        pred = predictions.median(dim=-1).values.unsqueeze(-1)
        return batch.transform['x'].inverse_transform(pred)

    def training_step(self, batch, batch_idx):
        self.define_scheduler_curriculum()
        noise = torch.randn_like(batch.x)
        x_itp = batch.x_interpolated
        y = prepare_y(batch)
        mask = batch.mask
        og_mask = batch.og_mask
        t = self.t_generator(batch.batch_size).to(batch.x.device)
        sigmas = self.scheduler.get_sigmas(t)
        sigmas_1 = self.scheduler.get_sigmas(t+1)

        sigmas_scaled, sigmas_1_scaled = rescale_vector(sigmas, y), rescale_vector(sigmas_1, y)

        # -------------- Forward student model --------------
        x_noisy_t_1 = y + noise * sigmas_1_scaled
        rng_state_cpu = torch.get_rng_state()
        if batch.x.device != torch.device('cpu'):
            rng_state_gpu = torch.cuda.get_rng_state()

        x_student = self.forward_cm(x_noisy_t_1, x_itp, mask, sigmas_1, sigmas_1_scaled)

        # -------------- Forward teacher model --------------

        if sigmas_scaled.max() > self.scheduler.sigma_min:
            with torch.no_grad():
                x_noisy_t = y + noise * sigmas_scaled

                torch.set_rng_state(rng_state_cpu)
                if batch.x.device != torch.device('cpu'):
                    torch.cuda.set_rng_state(rng_state_gpu)

                x_teacher = self.forward_cm(x_noisy_t, x_itp, mask, sigmas, sigmas_scaled)
        else:
            x_teacher = y


        # -------------- Compute loss --------------
        loss = self.loss_fn(x_student, x_teacher, og_mask, sigmas_scaled, sigmas_1_scaled)
        
        self.log_loss('train', loss, batch_size=batch.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self.generate_median_imputation(batch, n=10)
        loss = self.mse_loss(prediction, batch.y, batch.eval_mask)
        self.log_loss('val', loss, sync_dist=True, batch_size=batch.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        imputation = self.generate_median_imputation(batch)
        self.test_metrics.update(imputation, batch.y, batch.eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)

    def test_step_virtual_sensing(self, batch, masked_sensors):
        dict_sensors = {i:0 for i in masked_sensors}
        res = {
            'mae': dict_sensors,
            'mse': deepcopy(dict_sensors)
            }
        imputation = self.generate_median_imputation(batch)

        for sensor in masked_sensors:
            eval_mask = batch.eval_mask[:, :, sensor, :]
            y = batch.y[:, :, sensor, :]
            x = imputation[:, :, sensor, :]

            res['mae'][sensor] = self.mae_loss(x, y, eval_mask).cpu().item()
            res['mse'][sensor] = self.mse_loss(x, y, eval_mask).cpu().item()
            

        return res
