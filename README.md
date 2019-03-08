# mcv-m6-2019-team5

## Team members

- Alba María Herrera Palacio
- Jorge López Fueyo
- Nilai Sallent Ruiz
- Marc Núñez Ubach

## Running the code

To run this code you will need at least:

- Python 3.5

### Creating a virtual environment (Suggested)

Instead of installing all the dependencies to the global `python` installation, 
it is recommended to use a virtual environment.


```bash
pip install virtualenv         # if not installed already
python -v venv ./venv

source venv/bin/activate       # activate the environment
deactivate                     # deactivate the environment

```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the code

```bash
python src/main.py
```

## Directory structure

```
.
├── annotations                     # our annotations
├── datasets                        # datasets provided by the teachers
├── requirements.txt                # python dependencies
└── src
    ├── methods                     # pipelines used during the different weeks 
    ├── metrics                     # metrics to extract from the data
    ├── model                       # classes used to represent the domain model
    ├── operations                  # different operations to be used by the pipelines
    └── utils                       # miscellaneous utilities

```