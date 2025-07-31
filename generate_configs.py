# generate_configs.py
import IPython
import yaml
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

model_name_map = {
    "FACENETModel": "facenet",
    "iResNet100Model": "resnet101",
    "CLIPModel": "clip",

}

if __name__ == "__main__":
    CONFIG_PATH = Path(__file__).parent / "default.yaml"

    # Load the .yaml
    with CONFIG_PATH.open() as f:
        base_config = yaml.safe_load(f)

    param_grid = {
        "FFN": ["True", "False"],
        "model": ["FACENETModel", "iResNet100Model", "CLIPModel"],
        #"type": ["image", "embeddings"]

    }

    config_save_path = CONFIG_PATH.parent / "config"
    config_save_path.mkdir(exist_ok=True)

    for i, params in enumerate(ParameterGrid(param_grid)):
        config = deepcopy(base_config) # Copy template

        # Assign parameters
        config['model_configs']['model_name'] = params["model"]
        config['model_configs']['FFN'] = params["FFN"]
        #config['model_configs']['type'] = params["type"]
        #config['data_loader_configs']['mode'] = params["type"]
        config['data_loader_configs']['model_emb'] = model_name_map[params["model"]]

        name = f"{params['model']}_{params['FFN']}"
        out_file = config_save_path/ f"config_{name}.yaml"

        with open(out_file, "w") as f:
            yaml.dump(config, f)
