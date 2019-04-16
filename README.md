# mcv-m6-2019-team5

## Team members

- Alba María Herrera Palacio
- Jorge López Fueyo
- Nilai Sallent Ruiz
- Marc Núñez Ubach

## Report
-  {Link}[Docs/Report.pdf]

## Running the code

To run this code you will need at least:

- Python 3.5

### Creating a virtual environment (Suggested)

Instead of installing all the dependencies to the global `python` installation, 
it is recommended to use a virtual environment.


```
pip3 install virtualenv         # if not installed already
python3 -v venv ./venv

source venv/bin/activate       # activate the environment
deactivate                     # deactivate the environment
```

### Install dependencies

```
pip3 install -r requirements.txt
```

#### Pyflow

```
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
python setup.py install
cd ..
rm -rf pyflow
```

NOTE: remember to activate the environment before the install

If you get any problem compiling in windows, go to file `pyflow/src/project.h` and comment line 9: 

```c++
// #define _LINUX_MAC
```

### W3 usage

```
usage: main.py [-h] [-d] [-e EPOCHS]
               {fine_tune_yolo,off_the_shelf_yolo,off_the_shelf_ssd,siamese_train}
               [{siamese,overlap,kalman}]

Search the picture passed in a picture database.

positional arguments:
  {fine_tune_yolo,off_the_shelf_yolo,off_the_shelf_ssd,siamese_train}
                        Method to use
  {siamese,overlap,kalman}
                        Tracking method to use

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Show debug plots
  -e EPOCHS, --epochs EPOCHS
                        Number of train epochs

```

### W1-W2 usage

```
usage: main.py [-h] [--debug]
               {w2_adaptive,w2_nonadaptive,w2_soa,w2_nonadaptive_hsv,w2_adaptive_hsv,w2_soa_mod}

Search the picture passed in a picture database.

positional arguments:
  {w2_adaptive,w2_nonadaptive,w2_soa,w2_nonadaptive_hsv,w2_adaptive_hsv,w2_soa_mod}
                        Method to use

optional arguments:
  -h, --help            show this help message and exit
  --debug               Show debug plots
```

## Directory structure

```
.
├── config                          # configuration files used by neural networks
├── datasets                        # datasets provided by the teachers
├── requirements.txt                # python dependencies
├── src                             # Code for the third week
├── w1_w2                           # Code for the first two weeks
└── weights                         # Weights for different neural networks

```
