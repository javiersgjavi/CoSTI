from src.experiments.experiment import Experiment

class TestExperiment(Experiment):
    def run(self):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()

        # Test

        self.model.load_model(self.cfg.weights.path)
        self.model.freeze()

        results = self.trainer.test(self.model, self.test_dataloader)

        return results
