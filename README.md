
# A4S Evaluation - RAMIA Privacy Metric Implementation

**Name- Mohamed Ruzaik Fathima Sheema (025500264A) - MCYSD**

**GitLab Repository (Public):** https://gitlab.com/fathima.mohamed.001/a4s_ai_project_025500264a.git

------------------------------------------------------------------------------
## Project Overview

This project implements the **RAMIA (Range Membership Inference Attack)** privacy metric for evaluating privacy leakage in machine learning models, specifically image classification models trained on CIFAR-10.

------------------------------------------------------------------------------
## Available Metrics

### Privacy Metrics
- **Privacy Meter - RAMIA (Range Membership Inference Attack)**: Evaluates privacy leakage in image classification models by testing if data similar to training samples can reveal membership information.

------------------------------------------------------------------------------
## Installation Instructions

### Prerequisites
- Python 3.12 or higher.
- `uv` package manager (install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- WSL and Ubuntu installed.
- Have Jupyter Lab OR Jupyter Notebook installed.

---
### Setup Steps

1. **Clone the repository:**

- Open Ubuntu terminal and `git clone https://gitlab.com/fathima.mohamed.001/a4s_ai_project_025500264a.git`   OR 
- Download the zip file and extract it in your Linux/Ubuntu Folder.
- Open  the Ubuntu terminal and move into the folder `a4s_ai_project_025500264a-main` (it can be `a4s_ai_project_025500264a` if cloned).
- Your pwd should be `a4s_ai_project_025500264a-main` ( you should be inside this folder, do not move further into the `a4s-eval` folder or any other folder) 

- To verify:
Type `ls` in the terminal
If you see the below files and folders, you are in the correct directory.

```
'A4S CCLA (Entities).txt'      CONTRIBUTORS.md             LICENSE.md   config           scripts
'A4S ICLA (Individuals).txt'  'DOCUMENTATION NOTICE.txt'   NOTICE.md    docs             tasks
'A4S README.md'                Dockerfile                  README.md    notebooks        tests
 CONTRIBUTING.md               Dockerfile.dev              a4s_eval     pyproject.toml   uv.lock
```
---

2. **Install dependencies:**

After the cloning, in the Ubuntu terminal run the below commands:

`uv sync`

`uv sync --frozen --group dev --group test`



##### if you reach this point successfully your setup is correct and ready to execute!

---

3. **Running the RAMIA Privacy Metric**

Execute the RAMIA metric tests by running the following command in the terminal:

`uv run pytest tests/metrics/model_metrics/test_ramia.py -v`

**Expected output:**

Expected output will look similar to the one below:

```
================================================= test session starts ==================================================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0 -- /home/sheema_ruzaik/a4s_sheema/a4s-eval/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/sheema_ruzaik/a4s_sheema/a4s-eval
configfile: pyproject.toml
plugins: anyio-4.9.0, cov-6.2.1
collected 2 items

tests/metrics/model_metrics/test_ramia.py::test_ramia_metric_exists PASSED                                       [ 50%]
tests/metrics/model_metrics/test_ramia.py::test_ramia_metric_execution PASSED                                    [100%]

================================================== 2 passed in 25.86s ================================================== 
```


- `test_ramia_metric_exists` - Verifies metric is registered
- `test_ramia_metric_execution` - Runs RAMIA on ResNet20/CIFAR-10

Results are saved to: `tests/data/measures/privacy_meter_ramia.csv`

---


4. **Running the Notebook**

 Use either ONE  of the options below in the terminal :

#### Option 1: Using JupyterLab

1. **Launch JupyterLab:**

`uv run jupyter lab`

After you run the above command, you will see the following URLS in the terminal with your token as below. Please Ctrl+click on ONE of the URLs which will open Jupyter Lab in your browser
   
`http://localhost:8888/lab?token=<YOUR_TOKEN>`

`http://127.0.0.1:8888/lab?token=<YOUR_TOKEN>`

2. **Open and run:**
Navigate to `notebooks/ramia_visualization.ipynb` and run all cells.

#### Option 2: Using Jupyter Notebook

`uv run jupyter notebook`

After you run the above command, you will see the following URLS in the terminal with your token as below. Please Ctrl+click on ONE of the URLs which will open Jupyter Notebook in your browser

`http://localhost:8888/tree?token=<YOUR_TOKEN>`

`http://127.0.0.1:8888/tree?token=<YOUR_TOKEN>`

Navigate to `notebooks/ramia_visualization.ipynb` and run all cells.

#### Option 3: Execute Notebook from file explorer in VS Code in terminal

For this open the folder in VS code by typing the below command

`code .`

In the folder structure in VS code, navigate to `notebooks/ramia_visualization.ipynb` and run all cells.

---

## Experiment Details

### Model
- **Architecture:** ResNet20
- **Framework:** PyTorch (TorchScript JIT)
- **Path:** `tests/data/resnet20_cifar10_jit.pt`

### Dataset
- **Name:** CIFAR-10
- **Location:** `tests/data/cifar-10-batches-py/`
- **Training samples:** Batches 1-5 (50,000 images - same as model training)
- **Test samples:** Test batch (10,000 images - unseen by model)

### RAMIA Parameters
- `num_neighbors`: 5 (perturbed samples per image)
- `noise_scale`: 0.05 (Gaussian noise magnitude)
- `max_samples`: 500 per set (for performance)

### Output Metrics
The experiment produces visualizations and metrics including:
- Attack accuracy, precision, recall, F1 score
- Privacy leakage score (attack accuracy - 0.5)
- Confusion matrix
- RAMIA signal comparison (train vs test)

------------------------------------------------------------------------------
## Project Structure

a4s-eval/
├── a4s_eval/
│   └── metrics/
│       └── model_metrics/
│           └── privacy_meter_ramia.py    # RAMIA metric implementation
├── tests/
│   ├── metrics/
│   │   └── model_metrics/
│   │       └── test_ramia.py             # RAMIA tests
│   └── data/
│       ├── cifar-10-batches-py/          # CIFAR-10 dataset
│       ├── resnet20_cifar10_jit.pt       # Trained model
│       └── measures/
│           └── privacy_meter_ramia.csv   # Test results
├── notebooks/
│   └── ramia_visualization.ipynb         # Experiment notebook with visualizations
└── README.md                             # This file

------------------------------------------------------------------------------
## Troubleshooting

### Common Issues

**Issue:** `ERROR: file or directory not found: tests/metrics/model_metrics/test_ramia.py` when running tests
- **Solution:** You maybe in the `a4s-eval` folder. Navigate back to the parent folder (i.e. `a4s_ai_project_025500264a-main`) and then run the test again `uv run pytest tests/metrics/model_metrics/test_ramia.py -v` 

**Issue:** `ModuleNotFoundError` when running tests
- **Solution:** Run `uv sync --frozen --group dev --group test`

**Issue:** CIFAR-10 data not found
- **Solution:** Ensure `tests/data/cifar-10-batches-py/` exists with all batch files

**Issue:** Model file not found
- **Solution:** Verify `tests/data/resnet20_cifar10_jit.pt` exists

**Issue:** Notebook won't execute
- **Solution:** Make sure you've run the tests first to generate `privacy_meter_ramia.csv`

**Issue:** Jupyter asks for login/token
- **Solution:** Copy the URL with token from terminal, or run: `uv run jupyter lab` (recommended)


