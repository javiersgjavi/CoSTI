import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import open_dict

from src.data.traffic import MetrLADataset
from src.experiments.experiment import Experiment

from src.data.data_utils import get_similarity_metrla

class MissingExperiment(Experiment):
    def __init__(self, **kwargs):
        self.p_fault, self.p_noise = kwargs.pop('fault_noise')
        super().__init__(**kwargs)

        self.weights_path = self.cfg.weights.path

    def prepare_data(self):
        dm_params = {
            'batch_size': self.cfg.dataset.batch_size//len(self.device),
            'scale_window_factor': self.cfg.dataset.scale_window_factor,
            'p_fault': self.p_fault,
            'p_noise': self.p_noise,
            'point': True if self.cfg['dataset']['scenario'] == 'point' else False
        }

        self.dm = MetrLADataset(**dm_params).get_dm()
        self.dm_stride = MetrLADataset(stride='window_size', **dm_params).get_dm()
        self.adj_matrix = get_similarity_metrla(thr=0.1)

        self.dm.setup()
        self.dm_stride.setup()

        self.hist_patterns = None

        print(self.dm)

        with open_dict(self.cfg):
            self.cfg.dataset.time_steps = self.dm.window
            self.cfg.dataset.num_nodes = self.dm.n_nodes

        self.train_dataloader = self.dm.train_dataloader()
        self.val_dataloader = self.dm_stride.val_dataloader()
        self.test_dataloader = self.dm_stride.test_dataloader()

    def run(self):

        self.prepare_data()
        self.prepare_optimizer()

        self.prepare_model()
        print(self.trainer)

        # Test
        self.model.load_model(self.weights_path)
        self.model.freeze()

        results = self.trainer.test(self.model, self.test_dataloader)

        return results[0]

class MissingAverageExperiment:
    def __init__(self, dataset, cfg, optimizer_type, seed, epochs, accelerator='gpu', device=None, n=5):
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer_type = optimizer_type
        self.seed = seed
        self.epochs = epochs
        self.accelerator = accelerator
        self.device = device
        self.n = n
        self.folder = f'./metrics'
        self.file_results = f'{self.folder}/results_by_rate.csv'
        self.file_average_results = f'{self.folder}/results_average.csv'

        self.kwargs_experiment = {
            'dataset': self.dataset,
            'cfg': self.cfg,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'accelerator': self.accelerator,
            'device': self.device,
            'seed': seed,
        }

        self.missing_rates_point = { # rate: p_fault, p_noise
            0.1: (0, 0.0205),
            0.2: (0, 0.1295),
            0.3: (0, 0.2381),
            0.4: (0, 0.347),
            0.5: (0, 0.456),
            0.6: (0, 0.5649),
            0.7: (0, 0.6735),
            0.8: (0, 0.7822),
            0.9: (0, 0.8912),
        }

        self.missing_rates_block = { # rate: p_fault, p_noise
            0.1: (0.0007, 0),
            0.2: (0.002938, 0.05),
            0.3: (0.007516, 0.05),
            0.4: (0.012835, 0.05),
            0.5: (0.01923, 0.05),
            0.6: (0.027285, 0.05),
            0.7: (0.03786, 0.05),
            0.8: (0.05356, 0.05),
            0.9: (0.0819, 0.05),
        }

        self.init_result_folder()

    def init_result_folder(self):
        # Create results file if not exists
        os.makedirs(self.folder, exist_ok=True)
        if not os.path.exists(self.file_results):
            results = pd.DataFrame(columns=[
                'mae', 
                'mse', 
                'mre',
                'rate',
                'id'
                ])
            results.to_csv(self.file_results)

        # Create average results file if not exists
        if not os.path.exists(self.file_average_results):
            average_results = pd.DataFrame(columns=[
                'mae_mean',
                'mae_std',
                'mse_mean',
                'mse_std',
                'mre_mean',
                'mre_std',
                'rate'
            ])
            average_results.to_csv(self.file_average_results)
        

    def save_results(self, results):
        results_df = pd.read_csv(self.file_results, index_col='Unnamed: 0')
        results_df.loc[results_df.shape[0]+1] = [
            results['test_mae'], 
            results['test_mse'], 
            results['test_mre'], 
            results['rate'],
            results['id']
            ]
        results_df.to_csv(self.file_results)

    def average_results(self):
        average_results = pd.DataFrame(columns=[
                'mae_mean',
                'mae_std',
                'mse_mean',
                'mse_std',
                'mre_mean',
                'mre_std',
            ])
        
        results_by_experiment = pd.read_csv(self.file_results, index_col='Unnamed: 0')
        for rate in np.sort(results_by_experiment['rate'].unique()):
            results_rate = results_by_experiment[results_by_experiment['rate'] == rate]
            average_results.loc[rate] = [
                results_rate['mae'].mean(),
                results_rate['mae'].std(),
                results_rate['mse'].mean(),
                results_rate['mse'].std(),
                results_rate['mre'].mean(),
                results_rate['mre'].std(),
            ]
        
        average_results.to_csv(self.file_average_results)

    def run(self):
        for rate in np.arange(0.1, 1, 0.1):
            rate = np.round(rate, 2)
            for i in tqdm(range(self.n), desc=f'Running missing rate {rate}'):
                i = np.round(i, 2)
                results = pd.read_csv(self.file_results)
                done = results[(results['rate'] == rate) & (results['id'] == i)].shape[0] > 0
                if not done:
                    self.kwargs_experiment['seed'] = self.seed + i
                    self.kwargs_experiment['fault_noise'] = self.missing_rates_point[rate]

                    experiment = MissingExperiment(**self.kwargs_experiment)
                    results = experiment.run()
                    print(results)

                    results['rate'] = rate
                    results['id'] = i

                    self.save_results(results)

        self.average_results()