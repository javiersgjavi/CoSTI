import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.impute_experiment import ImputeExperiment

@hydra.main(config_name="base.yaml", config_path="../config/impute/")
def main(cfg: DictConfig):
    experiment = ImputeExperiment(
        dataset=cfg.dataset.name,
        cfg=cfg,
        optimizer_type=0,
        seed=42,
        epochs=200,
        accelerator='gpu',
        device=[0],
    )

    experiment.run()

if __name__=='__main__':
    main()