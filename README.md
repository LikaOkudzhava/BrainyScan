# brAIny

## Project Description

The aim of this project is to prepare, develop, and train a computer model based 
on convolutional neural networks (CNN) for analyzing and classifying brain MRI scans 
according to the severity of Alzheimer's disease.

## Dataset

[Alzheimer's Disease Multiclass Images Dataset](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented)

The data consists of MRI images. The data has four classes of images:

* Mild Demented
* Moderate Demented
* Non Demented
* Very Mild Demented

## How to try

### Docker

We've published pre-configured docker image on teh docker hub. It's name is _*n0imaginati0n/brainy:v1*_. For a moment it is proven to be run on linux, amd64 architecture and didn't run on Apple ARM (M1/M2/M3) CPU without emulation. To run it you need:

* [download and install docker desktop engine](https://www.docker.com/get-started/)
* if you had the Docker Desktop installed, search and choose the docker _*n0imaginati0n/brainy:v1*_ image, and then run it. don't forget to set port mapping from port 8080 of the container to the port 8080 of local system
* for command line:
    * check, that you can find the image
        ```shell
        docker search n0imaginati0n/brainy:v1
        ```
    * run it as seamless container
        ```shell
        docker pull n0imaginati0n/brainy:v1
        docker run -p 8080:8080 -d n0imaginati0n/brainy:v1
        ```
    * run it and see it's log report
        ```shell
        docker pull n0imaginati0n/brainy:v1
        docker run -p 8080:8080 -d -it n0imaginati0n/brainy:v1
        ```

After container run, connect to the service using your web-browser: [http://127.0.0.1:8080](http://127.0.0.1:8080)

### Manual run

It is possible to run the app locally without container. To do this:

* go to the app directory
    ```shell
    cd app/brainyscan/app
    ```

* install and set python 3.11.2 as a local version
    ```shell
    pyenv install 3.11.2
    pyenv local 3.11.2
    ```
* create a virtual python environment
    ```shell
    python -m venv .venv
    ```
*  and activate it
    Linux and MacOS
    ```shell
    source .venv/bin/activate
    ```

    Windows
    ```shell
    .venv\Scripts\activate
    ```

* install packages
    ```shell
    python -m pip install pip --upgrade
    pip install -r requirements.txt
    ```

* copy the previously saved model in to 'instance' directory as 'model.keras'
    
    ```shell
    cp <somewhere>/modle_file_name.keras instance/model.keras
    ```

* run the application

    ```shell
    flask --app wsgi run
    ```

    by default URL to connect will be shown on the screen. and it is usally [http://127.0.0.1:5000](http://127.0.0.1:5000)
