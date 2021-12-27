# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy


### Prerequisites
you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).
    
### Create and Activate Virtual Environment (Optional but highly recommended)

```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages. 

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv **venv** under folder /
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```

## Dependencies

The script has been tested running under **Python >= 3.6.0**, with the folowing packages installed:

```shell
pip install -r requirements.txt
```

### Questions

### About Tensorflow 2.0: https://github.com/hill-a/stable-baselines/issues/366

If you have questions regarding TensorFlow, note that tensorflow 2.0 is not compatible now, you may use

```bash
pip install tensorflow==1.15.4
 ```

If you have questions regarding Stable-baselines package, please refer to [Stable-baselines installation guide](https://github.com/hill-a/stable-baselines). Install the Stable Baselines package using pip:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Run DRL Ensemble Strategy
```shell
python run_DRL.py
```


