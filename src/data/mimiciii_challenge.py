import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.ops.imputation import add_missing_values
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.datasets.prototypes.tabular_dataset import TabularDataset

from src.data.transformations import ImputatedDataset, CustomTransform, CustomScaler

class DataChallenge:
    def __init__(self, val_ratio=0.1, test_ratio=0.1, path='../../data/mimic-iii_challenge/Dataset.csv'):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.path = path

    def load_csv(self):
        data = pd.read_csv(self.path)
        data = data.drop(columns=['Unnamed: 0'])
        patient_ids = data['Patient_ID'].unique()
        print(f'Num of patients: {len(patient_ids)}')

        return data, patient_ids

    def get_dict_patients_data(self, data, patient_ids):
        patients_data = {}
        for patient_id in tqdm(patient_ids, desc='Get patients data'):

            patient_data = data[data['Patient_ID'] == patient_id]
            patient_data = patient_data.drop(columns=['Patient_ID'])

            if patient_data['SepsisLabel'].sum() > 0:
                onset_row = patient_data[patient_data['SepsisLabel'] == 1].first_valid_index()
                patient_data = patient_data.loc[:onset_row]

            patients_data[patient_id] = patient_data

        return patients_data

    def pad_to_48h(self, patient):
        patient = patient.reset_index(drop=True)
        patient = patient.drop(columns=['Hour'])

        pad_size = 48 - patient.shape[0]
        patient.index = patient.index + pad_size
        patient = patient.reindex(np.arange(48))

        return patient

    def get_48h_window(self, patient):
        patient_window = patient.iloc[-48:]
        patient_window = patient_window.reset_index(drop=True)
        patient_window = patient_window.drop(columns=['Hour'])

        return patient_window

    def get_transform_data(self, patient_dict):
        for patient_id in tqdm(patient_dict.keys(), desc='Transform data'):
            patient = patient_dict[patient_id]
            if patient.shape[0] < 48:
                patient_dict[patient_id] = self.pad_to_48h(patient)
            else:
                patient_dict[patient_id] = self.get_48h_window(patient)

        data_list = list(patient_dict.values())

        data_array = np.stack(data_list)

        return data_array.transpose(0, 2, 1)

    def get_x_y_data(self, patients_array):
        x_dl = patients_array[:, :-1, :]

        y_data = patients_array[:, -1, :].max(axis=1)
        y_data = np.nan_to_num(y_data, 0)

        return [x_dl, y_data]

    def split_data(self, data):
        x_dl, y_data = data
        pos_val = int(x_dl.shape[0] * self.val_ratio)
        pos_test = int(x_dl.shape[0] * self.test_ratio)

        x_dl_train = x_dl[:-pos_val - pos_test]
        y_train = y_data[:-pos_val - pos_test]

        x_dl_val = x_dl[-pos_val - pos_test:-pos_test]
        y_val = y_data[-pos_val - pos_test:-pos_test]

        x_dl_test = x_dl[-pos_test:]
        y_test = y_data[-pos_test:]

        return [x_dl_train, y_train], [x_dl_val, y_val], [x_dl_test, y_test]

    def generate_data(self, folder_data):
        data_csv, patient_ids = self.load_csv()
        patients_dict_data = self.get_dict_patients_data(data_csv, patient_ids)
        patients_array = self.get_transform_data(patients_dict_data)
        data = self.get_x_y_data(patients_array)
        train, val, test = self.split_data(data)

        os.makedirs(f'{folder_data}/train/', exist_ok=True)
        os.makedirs(f'{folder_data}/val/', exist_ok=True)
        os.makedirs(f'{folder_data}/test/', exist_ok=True)

        np.save(f'{folder_data}/train/x.npy', train[0])
        np.save(f'{folder_data}/train/y.npy', train[1])

        np.save(f'{folder_data}/val/x.npy', val[0])
        np.save(f'{folder_data}/val/y.npy', val[1])

        np.save(f'{folder_data}/test/x.npy', test[0])
        np.save(f'{folder_data}/test/y.npy', test[1])

        return train, val, test

    def load_final_data(self, path):

        train = [np.load(f'{path}/train/x.npy'), np.load(f'{path}/train/y.npy')]

        val = [np.load(f'{path}/val/x.npy'), np.load(f'{path}/val/y.npy')]

        test = [np.load(f'{path}/test/x.npy'), np.load(f'{path}/test/y.npy')]

        return train, val, test

    def get_data(self):
        folder_data = f'../../data/mimic-iii_challenge'
        os.makedirs(folder_data, exist_ok=True)

        if not os.path.exists(f'{folder_data}/train/'):
            self.generate_data(folder_data)

        train, val, test = self.load_final_data(folder_data)

        return train, val, test
    

class MimicIIIChallengeDataset:
    def __init__(self, batch_size=4, p_fault=0., p_noise=0.25, **kwargs):

        self.seed = 42
        self.window_size = 48

        if not os.path.exists('../../data/mimic-iii_challenge/train/x.npy'):
            data_challenge = DataChallenge()
            _ = data_challenge.get_data()

        train_data = np.load('../../data/mimic-iii_challenge/train/x.npy')
        val_data = np.load('../../data/mimic-iii_challenge/val/x.npy')
        test_data = np.load('../../data/mimic-iii_challenge/test/x.npy')

        data = np.concatenate([train_data, val_data, test_data], axis=0)
        data = np.transpose(data, (0, 2, 1)).astype(np.float32).reshape(-1, data.shape[1])
        data_pd = pd.DataFrame(data)

        og_mask = ~np.isnan(data).astype(bool).reshape(data.shape[0], data.shape[1], 1)

        tabular_dataset = TabularDataset(
            target=data_pd,
            mask=og_mask,
        )

        dataset = add_missing_values(
            tabular_dataset,
            p_fault=p_fault,
            p_noise=p_noise,
            min_seq=self.window_size,
            max_seq=self.window_size * 2,
            seed=self.seed
        )

        connectivity = self._calculate_connectivity(data_pd)
        covariates = {'u': np.zeros((data_pd.shape[0], 2))}

        data[np.isnan(data).astype(bool)] = 0

        
        torch_dataset = ImputatedDataset(
            target=data.reshape(data.shape[0], data.shape[1], 1),
            og_mask=og_mask,
            mask=dataset.training_mask,
            eval_mask=dataset.eval_mask,
            covariates=covariates,
            transform=CustomTransform(),
            connectivity=connectivity,
            window=self.window_size,
            stride=self.window_size
        )

        print(f'Real missing values: {np.round((1 - np.mean(dataset.mask))* 100, 2)} %')
        print(f'Final missing values: {np.round((1 - np.mean(dataset.training_mask))* 100, 2)} %')

        scalers = {'target': CustomScaler()}


        idx_train_set = train_data.shape[0]
        idx_val_set = idx_train_set + val_data.shape[0]
        idx_test_set = idx_val_set + test_data.shape[0]

        print(f'Train set: {idx_train_set}')
        print(f'Val set: {idx_val_set}')
        print(f'Test set: {idx_test_set}')


        splitter = FixedIndicesSplitter(
            train_idxs=[i for i in range(idx_train_set)],
            val_idxs=[i for i in range(idx_train_set, idx_val_set)],
            test_idxs=[i for i in range(idx_val_set, idx_test_set)]
        )

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=batch_size,
            )
        
    def get_dm(self):
        return self.dm
    
    def _calculate_connectivity(self, data_pd):
        n_vars = data_pd.shape[1]

        edge_index = np.array([
            np.repeat(np.arange(n_vars), n_vars),
            np.tile(np.arange(n_vars), n_vars)
        ])

        correlation = data_pd.corr()

        edge_weights = []
        for i in range(edge_index.shape[1]):
            e1 = edge_index[0, i]
            e2 = edge_index[1, i]
            value = correlation.iloc[e1, e2]
            edge_weights.append(value)

        edge_weights = np.array(edge_weights).astype(np.float32)

        matrix = np.zeros((n_vars, n_vars))
        for i in range(edge_weights.shape[0]):
            matrix[edge_index[0, i], edge_index[1, i]] = edge_weights[i]

        np.save('../../data/mimic-iii_challenge/mimic_graph.npy', matrix)

        return edge_index, edge_weights
    
    def get_historical_patterns(self):
        self.dm.setup()
        train_loader = self.dm.train_dataloader()
        historical_patterns = []
        for batch in train_loader:
            historical_patterns.append(batch.mask)

        return torch.cat(historical_patterns, dim=0)