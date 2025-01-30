import torch
import numpy as np
from tsl.datasets import AirQuality
from tsl.data.datamodule import SpatioTemporalDataModule

from src.data.transformations import ImputatedDataset, CustomTransform, CustomScaler

class AirDataset:
    def __init__(self, stride=1, batch_size=4, scale_window_factor=1, test_months=(3, 6, 9, 12), masked_sensors=None):
        self.base_window = 36

        self.window_size = int(self.base_window * scale_window_factor)
        stride = self.window_size if stride == 'window_size' else stride

        og_mask = AirQuality(small=self.small, test_months=test_months, masked_sensors=masked_sensors).mask
        dataset = AirQuality(small=self.small, test_months=test_months, masked_sensors=masked_sensors)

        connectivity = dataset.get_connectivity(
            include_self=False,
            normalize_axis=1,
            layout="edge_index"
            )
        
        covariates = {'u': dataset.datetime_encoded('day').values}
        
        torch_dataset = ImputatedDataset(
            target=dataset.dataframe(),
            og_mask=og_mask,
            mask=dataset.training_mask,
            eval_mask=dataset.eval_mask,
            covariates=covariates,
            transform=CustomTransform(),
            connectivity=connectivity,
            window=self.window_size,
            stride=stride
        )

        print(f'Real missing values: {np.round((1 - np.mean(dataset.mask))* 100, 2)} %')
        print(f'Final missing values: {np.round((1 - np.mean(dataset.training_mask))* 100, 2)} %')

        scalers = {'target': CustomScaler()}

        splitter = dataset.get_splitter(method='air_quality', val_len=0.1)

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=batch_size,
            )
        
    def get_dm(self):
        return self.dm
    
    def get_historical_patterns(self):
        self.dm.setup()
        test_loader = self.dm.test_dataloader()
        historical_patterns = []
        for batch in test_loader:
            historical_patterns.append(batch.mask)

        return torch.cat(historical_patterns, dim=0)


class AQI36Dataset(AirDataset):
    def __init__(self, **kwargs):
        self.small = True
        super().__init__(**kwargs)