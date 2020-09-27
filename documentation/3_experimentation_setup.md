# 3. Experimentation setup

The experimentation setup is entirely based on Python. GhostNet (and some [other networks](https://github.com/pchaberski/cars/tree/documentation/models/architectures), which also can be used) implementation is written in PyTorch. Training process is orchestrated using `pytorch-lightning` package and controlled by parameters passed through YAML config file. Neptune experiment management tool (https://neptune.ai/) was used for experiment tracking. To build an environment for data preparation and model training Python `virtual env` utility was used. In addition to local training setup there is also a possibility to recreate the project environment and run training on Google Colab platform using a prepared Jupyter Notebook.

## 3.1. Project structure <a name="project-structure"></a>

Source code for the project is available in GitHub repository: https://github.com/pchaberski/cars. Repository contains the following elements:

* `documentation` - folder containing markdown files with project documentation, images, bibliography as a `.bib` file and some tools for document conversion
* `datasets` - Python package containing:
    * `stanford_data.py` module implementing class for Stanford Cars data loading and preprocessing
    * `stanford_data_module.py` - module implementing `LightningDataModule` defining data loaders for main training `LightnigModule`
    * `stanford_utils.py` - utility to process raw files downloaded from dataset webpage to be suitable for training and validation
* `models` - Python package containing:  
    * `architectures` - folder with modules implementing `GhostNet` and several other architectures that were briefly tested during initial stage of the project (`SqueezeNet`, `SqueezeNext`, `EfficientNet`, `MobileNet-V2`, `ShuffleNet`, `HarDNet`)
    * `arch_dict.py` - module with a dictionary of architectures that can be used in experiments
    * `net_module.py` - module containing main `LightningModule` used for network training and evaluation
    * `label_smoothing_ce.py` - implementation of Label Smoothing Cross Entropy loss function [[15]](5_references.md#Poulopoulos2020)
* `utils` - Python packages with utilities for configuration parsing, logging and execution time measurement
* `config_template.yml` - YAML configuration file template; it is supposed to be filled and saved as `config.yml` to allow controlling training settings (mostly data preprocessing settings and model hyperparameters) without interference with source code
* `prod_requirements.txt` - list of external PyPI Python packages to be included in `virtual env` to run the training
* `dev_requirements.txt` - list of additional PyPI Python packages that were used during development and results postprocessing
* `prepare_stanford_dataset.py` - executable Python script that prepares raw files from dataset website to the form suitable for training and validation
* `train.py` - main executable Python script for running experiments
* `train_colab.ipynb` - Jupyter Notebook that can be used to recreate local working environment on Google Colab and run `train.py` remotely

## 3.2. Working environment <a name="working-environment"></a>

## 3.3. Configuration <a name="configuration"></a>

## 3.4. Experiment tracking <a name="experiment-tracking"></a>  

