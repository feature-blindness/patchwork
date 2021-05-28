# patchwork
Feature-blindness in humans and CNNs.


*****************
**ADD GIFS HERE**
*****************

## Datasets
The folder `datasets` contains an example (single seed) of four type of datasets:
- **Patch**: [`datasets/stim_dpatch_invalid_jit3`] A single patch and overall shape are predictive of category membership.
- **Segment**: [`datasets/stim_dcol_invalid_jit3/`] The colour of one segment and overall shape are predictive of category membership.
- **Size**: [`datasets/stim_dcell_size_invalid_jit3/`] The average size of patches and overall shape are predictive of category membership.
- **Color**: [`datasets/stim_dunicol_jit3/`] The average size of patches and overall shape are predictive of category membership.

Each of these folders contains `p0` and `p20` subfolders. These correspond to the conditions where shape is predictive for 100% of training stimuli and 80% of training stimuli, respectively.


## CNN Simulation
The folder `cnn_simulations` contains code to generate datasets and run the CNN simulations.

### Requirements
Please see `patchwork/cnn_simulations/requirements.txt` for the python packages used to run this code.

### Usage
- Generate data by running `gen_stim.py` (e.g. `$ python gen_stim.py`). Data will be generated in the `patchwork/cnn_simulations/data` subfolder. To use these data for the simulations, it will need to be moved to the `patchwork/datasets` folder.
    - Note 1: By default this will generate the _Segment_ condition. To generate data from a different condition, change the variable `experiment` in the code.
    - Note 2: By default the script will generate 10 seeds (10 independent training sets) with 2000 training trials for each category and 500 test trials for each category in each test condition. Change the variables `nseed`, `ntrain` and `ntest` to change this behaviour.

- Train (and test) the CNN by running `sim_cnn.py` (e.g. `$ python sim_cnn.py`). By default this script will train and test multiple seeds under each of the four conditions (_Patch_, _Segment_, _Size_ and _Colour_). This can be changed by changing the variable `test_set` in the script.
    - Note 1: Th default network is `ResNet50`. This can be changed by changing the variable `model_name`.
    - Note 2: The default training is for 20 epochs (networks are pretrained on ImageNet). This can be adjusted by changing the variable `num_epochs`.
    - Note 3: The script trains a different instantiation of the network on each seed of dataset present in the `patchwork/datasets` folder. By default, the script expects only one seeds. (The results shown in the manuscript were generated for 10 seeds). This can be adjusted by changing the variable `nseeds`.
    - Note 4: By default the script analyses data for the condition where shape predicts category in 80% of the training trials. To analyse data in the condition where shape is predictive for 100% training trials, change the variable `valid_value`.
    - Note 5: The script finishes by analysing the results and generating a plot in the `results` folder.


## Human Experiments

### Stimuli

### Script in PsychoPy

### Data and Analysis