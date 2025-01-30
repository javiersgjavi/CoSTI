import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.experiment import AverageExperiment

@hydra.main(config_name="base.yaml", config_path="../config/base/")
def main(cfg: DictConfig):

    experiment = AverageExperiment(
        cfg=cfg,
        device=0,
        seed=cfg.seed,
        epochs=cfg.epochs,
        n=cfg.n_experiments,
        dataset=cfg.dataset.name,
        accelerator=cfg.accelerator,
        optimizer_type=cfg.optimizer_type,
    )

    experiment.run()

if __name__=='__main__':
    main()