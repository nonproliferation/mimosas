# MIMOSAS: Multisource Input Model Output Security Analysis Suite v1.0.0-release.1

A supervised machine learning pipeline, MIMOSAS (Multisource Input Model Output Security Analysis Suite), has been developed for classification of multimodal data to inform nuclear security and proliferation detection scenarios. MIMOSAS provides an end-to-end data processing workflow, from data ingestion and pre-processing to model training and test set classification. The pipeline is specified via an input deck, making workflow customization effortless, and the framework is modular allowing for the easy addition of new learning algorithms.

In the current build, the user selects from decision tree, random forest, and feed-forward neural network classifiers to train customizable models with built-in cross validation methods for hyperparameter optimization. Trained model outputs are stored with the associated metadata for rapid deployment. These can be applied in supervised classification to assess previously unseen data or for further training as new observations are added to the existing data set. MIMOSAS provides the capability to fuse a wide range of data sources (e.g., radiation, environmental, acoustic, seismic, imagery, etc.) to make, confirm, and correlate machine learning predictions for nuclear security applications.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

MIMOSAS requires Python 3, among other scientific computation packages. To install the latest version of Python 3, see https://www.python.org/downloads/. Once Python 3 is installed, install additional packages using pip:

```
pip install numpy pandas scikit-learn pytorch scipy matplotlib ipython jupyter sympy nose
```
If prefered, Anaconda, a Python environment manager, may also be used. After activating your Anaconda environment, use conda instead of pip.

### Usage

Run using default.config config file. If default.config does not exist, automatically generate using source.config.
```
python main.py
```

Run using custom.config config file
```
python main.py --config custom.config
```

Print out license to console
```
python main.py --license
```

### Architecture

Analysis Refactor 2 is designed to be modular. It reads configuration inputs from any specified config file, and runs on any number of algorithms. Currently, random forest and decision tree are available. On running the algorithms, the user also has the option to run in Train&Test or Validation mode. Each algorithm has a Train and Test function for these purposes. The Test function is reused for Validation, as the implementation of the 2 are identical. However, it is important to note the operational differences between Test and Validation.

Adding additional algorithms is quite simple - it only involves implementing the train and test functions of the algorithm, adding a run function in main.py, and adding necessary algorithm options in source.config (where default config files are generated from). Parameter parsing is handled automatically, although one may find it beneficial to see how parameters are retrieved post parsing.

### Config file options

Config files are broken down into 4 major components - [MAIN] [ALGORITHM OPTIONS] [TRAINING_DATA] [EVALUATION_DATA], each serving a different function.

    [MAIN] - options that govern the overall operation of the analysis suite (ex. verbose mode, plotting enabled, etc.)
    [ALGORITHM OPTIONS] - each algorithm has its own section (ex. [DECISION_TREE]) that contains options specific to that algorithm. If an algorithm section is excluded entirely (ex. doesn't have [RANDOM_FOREST]), then that algorithm is not run.
    [TRAINING_DATA] - data cleaning and split options for training. certain options (ex. random shuffling) can be ignored for time-series algorithms.
    [EVALUATION_DATA] - data cleaning and split options for testing and evaluation. certain options (ex. random shuffling) can be ignored for time-series algorithms.

## Authors

* **Jared Zhao**
* **Bethany L. Goldblum**
* **Christopher Stewart**
* **Alicia Ying-Ti Tsai**
* **Shruthi Chockkalingam**
* **Pedro Vicente Valdez**

## License

See the [LICENSE.txt](LICENSE.txt) file for details
