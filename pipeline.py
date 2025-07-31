# Repo Scripts
from matplotlib.testing.jpl_units import Epoch

import dataloader as dl_script
import architectures as ma_script

# Frameworks
import torch
from plotly.data import experiment
from torch import nn
from torch.nn import Conv2d, Softmax
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from torchvision.models import resnet101
import torch.optim as optim
import IPython
import pexpect
import socket
from rich.traceback import install
install()
from pathlib import Path


from architectures import *
import wandb
import time
from datetime import datetime

from tqdm import tqdm
import argparse
import yaml
import logging

from dataloader import MyDataLoader
from loss_wrappers import MultiHeadLossWrapper

class FileLogger:
    def __init__(self, experiment_name):
        # Files
        log_file = f"training_log.txt"
        self.experiment_name = experiment_name
        self.model_dir_path = experiment_name.split("__")[0]

        # Paths
        self.store_full_path = Path(__file__).parent / "models" / self.model_dir_path
        self.store_full_path.mkdir(parents=True, exist_ok=True)

        # Logging object
        self.logger = logging.getLogger(self.model_dir_path)
        self.logger.setLevel(logging.DEBUG)

        # File Handler
        fh = logging.FileHandler(str(self.store_full_path / log_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)

    def __call__(self, message):
        self.logger.info(message)


    def new_file_print(self):
        message = """
        
        ================================================
        NEW JOB RUNNING
        ================================================
        
        """
        self.logger.info(message)


class TrainingPipeline(nn.Module):
    def __init__(self, configs:dict) -> None:
        super(TrainingPipeline, self).__init__()

        # Variables Initialization
        self.run = None
        self.config = configs

        # File Logger + wandb Configuration
        self.experiment_name = self.get_experiment_name()
        self.logger = FileLogger(self.experiment_name)
        self.logger.new_file_print()
        self.wandb_logger(experiment_name=self.experiment_name)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger(f"Device being used: {self.device}")

        ## DataLoaders
        self.train_data_loader = MyDataLoader(
            dataset=getattr(dl_script, self.config["data_loader_configs"]["data_loader_name"])(
                **self.config["data_loader_configs"],
                state="train",
                file_logger=self.logger,
                device = self.device
            ),
            batch_size=self.config["model_configs"]["batch_size"],
            **configs['data_loader_configs']
        )
        self.valid_data_loader = MyDataLoader(
            dataset=getattr(dl_script, self.config["data_loader_configs"]["data_loader_name"])(
                state="valid",
                file_logger=self.logger,
                **self.config["data_loader_configs"],
                device=self.device
            ),
            batch_size=self.config["data_loader_configs"]["val_batch_size"],
            **configs['data_loader_configs']
        )

        self.test_data_loader = MyDataLoader(
            dataset=getattr(dl_script, self.config["data_loader_configs"]["data_loader_name"])(
                state="test",
                file_logger=self.logger,
                **self.config["data_loader_configs"],
                device=self.device
            ),
            batch_size=self.config["data_loader_configs"]["test_batch_size"],
            **configs['data_loader_configs']
        )

        # Model + Loss Wrapper
        self.model = getattr(ma_script, configs['model_configs']['model_name'])(
            experiment_name=self.experiment_name,
            file_logger=self.logger,
            device=self.device,
            **configs['model_configs'],
        )
        self.model_loss = MultiHeadLossWrapper(base_model=self.model)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["model_configs"]["learning_rate"]
        )

    def get_experiment_name(self):
        model_configs = self.config["model_configs"]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        model_name = f'{model_configs["model_name"]}-{model_configs["FFN"]}-p{[model_configs["p_dropout"]]}'
        return model_name + "__" + current_time

    def wandb_logger(self, experiment_name):
        wandb.login(key=self.config["wandb_params"]["wandb_key"], force=True, verify=True, relogin=True)
        self.run = wandb.init(
            project=self.config["wandb_params"]["project"],
            entity=self.config["wandb_params"]["entity"],  # Important to be the team name; otherwise it will give 403 error
            name=experiment_name,  # Experiment_name
            config=self.config
        )
        assert self.run is wandb.run

    def eval_model(self, data_loader, prefix: str):
        all_preds = {key: [] for key in self.model_loss.head_names}
        all_labels = []

        with torch.no_grad():
            for x_batch, y_true in tqdm(data_loader, desc=f"Evaluating {prefix} Batches", unit="batch"):
                y_pred = self.model_loss.base_model.predict(x_batch)

                # Save to correct dict
                for head in self.model_loss.head_names:
                    all_preds[head].append(y_pred[head])

                all_labels.append(y_true)

        # Stack all Preds & Labels
        for head in self.model_loss.head_names:
            all_preds[head] = torch.cat(all_preds[head], dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return self.model_loss.base_model.compute_metrics(all_preds, all_labels,
                                                          head_names=self.model_loss.head_names,
                                                          prefix=prefix)

    def run_training(self):
        for epoch in range(1, self.config["model_configs"]["epochs"] + 1):
            train_loss = 0
            epoch_start_time = time.time()

            with tqdm(total=len(self.train_data_loader), desc="Training Batches", unit="batch") as pbar:
                for b, (x_train, y_labels) in enumerate(self.train_data_loader): # Batches iteration

                    self.optimizer.zero_grad()
                    loss = self.model_loss(x_train, y_labels)

                    # Gradient
                    loss.backward() # Calculate Gradients, Store Gradient in .grad attribute in each parameter.
                    self.optimizer.step() # Update parameters. Applies the BackPropagation

                    train_loss += loss.item() # The loss within an epoch is the sum of the total loss of the batch
                    pbar.update(1)

            self.logger(f"Epoch {epoch} -> Loss: {loss.item()}")

            #if epoch % 5 == 0 and epoch > 4:
            if epoch % 1 == 0:
                self.model_loss.base_model.save_weights(epoch=epoch)

            train_metrics = self.eval_model(self.train_data_loader, prefix="train")
            val_metrics = self.eval_model(self.valid_data_loader, prefix="valid")
            test_metrics = self.eval_model(self.test_data_loader, prefix="test")

            epoch_metrics = {
                "Epoch": epoch,
                "Epoch Time": time.time() - epoch_start_time,
                "Train_loss": train_loss,
            }
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(val_metrics)
            epoch_metrics.update(test_metrics)
            wandb.log(epoch_metrics)

        self.logger(f"✅ Training Complete")
        wandb.finish()




def args_parser():
    parser = argparse.ArgumentParser(description="Run Training Pipeline")
    parser.add_argument(
        "-y", "--yaml_path", type=str, required=True,
        help="Path to the configuration YAML file."
    ),
    parser.add_argument(
        "-f", "--fraction", type=float, required=False, default=1.0,
        help="Fraction of training data to use.")
    parser.add_argument(
        "-p", "--p_dropout", type=float, required=True,
        help="Dropout Layer probability.")
    return parser.parse_args()

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

def main(args):
    config = load_yaml(args.yaml_path)
    config['data_loader_configs']['fraction'] = args.fraction
    config['model_configs']['p_dropout'] = args.p_dropout

    training_pipeline = TrainingPipeline(configs=config)
    training_pipeline.run_training()



if __name__ == "__main__":
    main(args=args_parser())

    #FileLogger(experiment_name = "experiment_test")

    # run it with fish shell: for f in config/*.yaml; do python pipeline.py -y "$f"; done
