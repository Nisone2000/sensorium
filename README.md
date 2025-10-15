This repository builds on top of [SENSORIUM 2022 codebase](https://github.com/sinzlab/sensorium).
Clustering loss is effectivelly added in the `standard_trainer` function in `sensorium/training/trainers.py`, all the input arguements starting from `include_kldivergence` relate to DECEMber.

Example usage
```
standard_trainer(
    ...
    include_kldivergence=True, # turns on DECEMber
    cluster_number=10, # specifies number of clusters
    dec_starting_epoch=5, # defines the pretraining duration (in epochs)
    base_multiplier=4e3, # defines clustering strength \beta
    **kwargs,
)
```


## Starter-kit (copied from [SENSORIUM 2022 repo](https://github.com/sinzlab/sensorium))

Below we provide a step-by-step guide for getting started with the competition.

### 1. Pre-requisites
- install [**docker**](https://docs.docker.com/get-docker/) and [**docker-compose**](https://docs.docker.com/compose/install/)
- install git
- clone the repo via `git clone https://github.com/sinzlab/sensorium.git`

### 2. Download neural data

You can download the data from [https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022) and place it in `sensorium/notebooks/data`.
**Note:** Downloading the files all at once as a directory does lead to unfortunate errors. Thus, all datastes have to be downloaded individually.

### 3. Run the example notebooks

**Start Jupyterlab environment**
```
cd sensorium/
docker-compose run -d -p 10101:8888 jupyterlab
```
now, type in `localhost:10101` in your favorite browser, and you are ready to go!


### **Competition example notebooks**
We provide notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
<br>[**Dataset tutorial**](notebooks/dataset_tutorial/): Shows the structure of the data and how to turn it into a PyTorch DataLoader.
<br>[**Model tutorial**](notebooks/model_tutorial/): How to train and evaluate our baseline models.
<br>[**Submission tutorial**](notebooks/submission_tutorial/): Use our API to make a submission to our competition.
