homework1
==============================

ML-project for ML in production homework 1
Python 3.8

After cloning repo don't forget to mark "ml_project" directory as Source Root

Installation (for Windows):  

    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install -r requirements.txt

Installation (for Linux):  

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Project summary
------------
In project used dataset [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) from kaggle.com

* [Hydra](https://hydra.cc/) - for configuration
* [pytest](https://docs.pytest.org/) - for tests

Commands
------------
Run model training pipeline
`python -m src.models.train_model`

Make predictions
`python -m src.models.predict_model --model_path _path_to_model.pkl_
                                    --data_path _path_to_data.scv_
                                    --out_path _path_to_output_file.scv_`


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │ 
    ├── configs            <- Configurations for project and logging
    │ 
    ├── logs               <- Generated log files
    │ 
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │  
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │ 
    ├── predictions        <- Generated files with predictions
    │    
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   
    │
    └── tests            <- tests for project modules


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
