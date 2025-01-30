import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.k_test_experiment import KTestAverageExperiment

@hydra.main(config_name="metr-la_point.yaml", config_path="../config/k_test/")
def main(cfg: DictConfig):

    experiment = KTestAverageExperiment(
        cfg=cfg,
        device=0,
        seed=cfg.seed,
        epochs=cfg.epochs,
        n=cfg.n_experiments,
        dataset=cfg.dataset.name,
        accelerator=cfg.accelerator,
        optimizer_type=cfg.optimizer_type,
        test_sigmas=cfg.test_sigmas
    )

    experiment.run()

if __name__=='__main__':
    main()