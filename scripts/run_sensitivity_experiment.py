import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.sensitivity_experiment import MissingAverageExperiment

@hydra.main(config_name="metr-la_point.yaml", config_path="../config/sensitivity/")
def main(cfg: DictConfig):

    experiment = MissingAverageExperiment(
        cfg=cfg,
        device=0,
        seed=cfg.seed,
        epochs=cfg.epochs,
        n=cfg.n_experiments,
        dataset=cfg.dataset.name,
        accelerator=cfg.accelerator,
        optimizer_type=cfg.optimizer_type
    )

    experiment.run()

if __name__=='__main__':
    main()