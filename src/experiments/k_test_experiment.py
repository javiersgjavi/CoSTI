import os
import time
import numpy as np
import pandas as pd
from src.experiments.experiment import Experiment, AverageExperiment

class KTestExperiment(Experiment):
    def __init__(self, test_sigmas, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_sigmas = test_sigmas
        self.weights = weights

    def run(self):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()

        # Test
        self.model.change_sigmas_predict(self.test_sigmas)
        self.model.load_model(self.weights)
        self.model.freeze()

        test_start = time.time()
        results = self.trainer.test(self.model, self.test_dataloader)
        test_end = time.time()

        training_time = 0
        testing_time = test_end - test_start

        results[0]['training_time'] = training_time
        results[0]['testing_time'] = testing_time

        return results[0]
    
class KTestAverageExperiment(AverageExperiment):
    def __init__(self, test_sigmas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs_experiment['test_sigmas'] = test_sigmas
        self.weight_paths = None

    def define_path_weights(self):
        base_path = f'''{self.kwargs_experiment['cfg']['weights']['path']}'''
        weight_files = np.sort(os.listdir(f'{base_path}'))
        weight_folders = [os.listdir(f'{base_path}/{n_experiment}/')[0] for n_experiment in weight_files]
        weights_paths = []
        for n_experiment, ckpt_file in zip(weight_files, weight_folders):
            res = f'{base_path}/{n_experiment}/{ckpt_file}'
            print(res)
            weights_paths.append(res)

        self.weight_paths = np.sort(weights_paths)



    def run(self):
        self.define_path_weights()
        n_done = pd.read_csv(f'{self.folder}/results_by_experiment.csv').shape[0]
        for i in range(n_done, self.n):
            self.kwargs_experiment['weights'] = self.weight_paths[i]
            self.kwargs_experiment['seed'] = self.seed + i
            experiment = KTestExperiment(**self.kwargs_experiment)
            results = experiment.run()
            self.save_results(results, i)

        self.average_results()
