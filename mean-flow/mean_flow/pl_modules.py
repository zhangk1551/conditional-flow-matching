import torch
import pytorch_lightning as pl

from torchcfm.utils import sample_8gaussians

from mean_flow.flow_models import MeanFlow
from mean_flow.models import MLP
from mean_flow.utils import plot_samples


class MeanFlowModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.eval_gap = self.config.evaluation.num_train_epoch_per_evaluation

        self.net = MLP(dim=2)

        self.mean_flow = MeanFlow(config.mean_flow, self.net)


    def training_step(self, batch, batch_idx):
        e, x = batch
        e, x = e.to(self.device), x.to(self.device)
        loss = self.mean_flow.loss(e, x)
        self.log_dict({"train_loss": loss})
        return loss


    def _set_model_eval(self):
        self.mean_flow.eval()
        self.net.eval()


    def _set_model_train(self):
        self.mean_flow.train()
        self.net.train()


    def on_train_epoch_end(self):
        if self.current_epoch % self.eval_gap == 0:
            self._set_model_eval()
            with torch.no_grad():
                self.visualize_samples()
            self._set_model_train()


    def visualize_samples(self):
        steps = [1, 2, 10]
        noises = sample_8gaussians(1024).to(self.device)
        plot_samples_images = []
        for step in steps:
            samples = self.mean_flow.sample(noises, step)
            plot_samples_images.append(plot_samples(noises.cpu().detach().numpy(), samples.cpu().detach().numpy()))
        self.logger.log_image(key=f"samples",
                              images=plot_samples_images,
                              caption=[f"{i}-step sampling" for i in steps])


    def configure_optimizers(self):
        lr = self.config.trainer.lr
        optim_method = self.config.trainer.optim_method
        if optim_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if optim_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer
