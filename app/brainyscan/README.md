# BraAIny Application

## System requirements
As the python version should match the version on Docker image, the used python should have version 3.11.2
It would be better to organize separate virtual python environment for this project as it can be different from the one used for a main project

```shell
pyenv install 3.11.2
pyenv local 3.11.2
```

```shell
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

## Start backend in terminal

Before start one need to copy selected model to the _instance_ directory

```shell
cp ../../models/models/XceptionBest_model_fittable.keras ./instance/model.keras
```

The backend can be started from this directory by

```shell
flask --app wsgi run
```

The backend app will print in the console URL used to connect to. Usually it is [BrAIny http://127.0.0.1:5000](http://127.0.0.1:5000)

Closing terminal will close the server instance. The backend instance keeps all the history and statistics in the memory, means it will be lost after the restart

