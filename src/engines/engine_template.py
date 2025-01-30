from schedulefree import AdamWScheduleFree
from tsl.engines.imputer import Imputer
from src.data.data_utils import create_interpolation, redefine_eval_mask

class CustomEngine(Imputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx)
        self.mpg.update_mask(batch)
        batch = create_interpolation(batch)
        batch = redefine_eval_mask(batch)

    def on_validation_batch_start(self, batch, batch_idx: int) -> None:
        super().on_validation_batch_start(batch, batch_idx)
        batch = create_interpolation(batch)
        batch = redefine_eval_mask(batch)

    def on_test_batch_start(self, batch, batch_idx: int) -> None:
        super().on_test_batch_start(batch, batch_idx)
        batch = create_interpolation(batch)
        batch = redefine_eval_mask(batch)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().train()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().eval()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().eval()
    
    def log_metrics(self, metrics, **kwargs):
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        self.log(
            name + '_loss',
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            **kwargs
        )