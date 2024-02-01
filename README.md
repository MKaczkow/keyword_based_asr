# Keyword Based ASR

Repo for implementation of keyword-based ASR system

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Setup
* create virtual enviroment and install requirements from `requirements.txt`
* `NOTE`: NeMo toolkit is not supported on Windows, so WSL or UNIX-based OS is required, see [the docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/starthere/intro.html) or [github](https://github.com/NVIDIA/NeMo)
* minimal, necessary data is already in the repo, but to reproduce training process and / or test other keywords you need to download full datasets from [this link](https://wutwaw-my.sharepoint.com/:u:/g/personal/01144023_pw_edu_pl/ESo3c_M06qxJlcUMNz32t_kBsfwwGn-CiQlNTg_7pcgw5w?e=L73fIA) (if the link doesn't work, please contact me via email: `maciej.kaczkowski.stud@pw.edu.pl`) and put them in `Data` directory (check [Data README](Data/README.md) for the structure)
* modify [config file](Utils/config.py) to match your setup (all paths with suffix `DATA_DIR` should be changed to match your setup)

## Repository structure
* [Data](./Data/) - directory with data
* [Utils](./Utils/) - directory containing utility scripts and config files
* [Models](./Models/) - directory containing trained models' weights
* notebooks in root directory
  * main
    * `demo.ipynb` - notebook demonstrating dual-model keyword-based speaker recognition system
    * `speaker-recognition.ipynb` - notebook with speaker recognition model training and evaluation
    * `keyword-recognition.ipynb` - notebook with keyword spotting model demonstration and evaluation
  * suplementary
    * `get-data.ipynb` - check audio files metadata
    * `visualize-spectrograms.ipynb` - visualize spectrograms of audio files
    * `play-sound.ipynb` - sanity-check audio files
