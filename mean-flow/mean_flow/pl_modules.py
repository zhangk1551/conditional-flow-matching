import torch
import pytorch_lightning as pl

from torchcfm.utils import sample_8gaussians
from torchvision import transforms


from mean_flow.configs import ModelType, DataType
from mean_flow.flow_models import MeanFlow
from mean_flow.models import MLP, UNet
from mean_flow.utils import plot_samples, plot_images_grid


class MeanFlowModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.eval_gap = self.config.evaluation.num_train_epoch_per_evaluation

        if config.model.model_type == ModelType.U_NET:
            self.net = UNet(image_size=32,
                                  in_channels=1,
                                  model_channels=32,
                                  out_channels=1,
                                  num_res_blocks=1,
                                  attention_resolutions=[],
                                  channel_mult=(1, 2, 4))
        else:
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
                if self.config.data.data_type == DataType.MNIST:
                    self.visualize_mnist_samples()
                else:
                    self.visualize_samples()
            self._set_model_train()


    def visualize_samples(self):
        steps = [1, 2, 10]
        noises = sample_8gaussians(1024).to(self.device)
        plot_samples_images = []
        for step in steps:
            samples = self.mean_flow.sample(noises, step)
            plot_samples_images.append(plot_samples(noises.cpu().detach().numpy(), samples.cpu().detach().numpy()))
        self.logger.log_image(key="samples",
                              images=plot_samples_images,
                              caption=[f"{i}-step sampling" for i in steps])


    def visualize_mnist_samples(self) -> None:
        steps: list[int] = [1, 2, 10]
        noises = torch.randn(4, 1, 32, 32).to(self.device)
#        noises = torch.randn(4, 1, 28, 28).to(self.device)
        reverse_transform = transforms.CenterCrop(28)
        image_plots = []

        for step in steps:
            samples = reverse_transform(self.mean_flow.sample(e=noises, step=step))
#            samples = self.mean_flow.sample(e=noises, step=step)
            images = samples.cpu().detach().numpy()

            image_plots.append(plot_images_grid(images, num_rows=2, num_cols=2))
        self.logger.log_image(
            key="samples",
            images=image_plots,
            caption=[f"{i}-step sampling" for i in steps]
        )


    def configure_optimizers(self):
        lr = self.config.trainer.lr
        optim_method = self.config.trainer.optim_method
        if optim_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if optim_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer
