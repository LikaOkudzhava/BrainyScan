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

We've published pre-configured docker image on teh docker hub. it's name is _*n0imaginati0n/brainy*_. To run it you need:

* [install docker engine](https://docs.docker.com/desktop/)
* if you had the Docker Desktop installed, search and choose the docker _*n0imaginati0n/brainy*_ image, and then run it. don't forget to set port mapping from port 8080 of the container to the port 8080 of local system
* for command line:
    * check, that you can find the image
        ```shell
        docker search n0imaginati0n/brainy
        ```
    * run it as seamless container
        ```shell
        docker pull n0imaginati0n/brainy
        docker run -p 8080:8080 -d n0imaginati0n/brainy
        ```
    * run it and see it's log report
        ```shell
        docker pull n0imaginati0n/brainy
        docker run -p 8080:8080 -d -it n0imaginati0n/brainy
        ```

After container run, connect to the service using your web-browser: [http://127.0.0.1:8080](http://127.0.0.1:8080)


