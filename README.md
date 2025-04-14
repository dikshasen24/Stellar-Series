# Stellar Flares Detection from Photometric Data Using Clustering and Machine Learning

This repository contains our work on using **unsupervised clustering** and **predictive modeling** to detect stellar flares from **TESS photometric time-series data**. Our goal is to develop an approach that can accurately identify stellar flares and predict future occurrences. We preprocess flux data from **TIC 0131799991**, perform exploratory data analysis (EDA), and apply clustering techniques for initial flare detection. A predictive model is then trained using the detected flare data.

## **Code and Workflow**

- **Data-preprocessing and EDA**: `EDA.ipynb` handles missing value imputation and performs exploratory data analysis to understand the structure and trends in the stellar time series.
- **Flare Detection**: `DBSCAN.ipynb` applies DBSCAN clustering to identify potential stellar flares based on flux features.
- **Flare Injection & Recovery**: `Simulations.ipynb` simulates synthetic flares and injects them into the light curve. Detection accuracy is evaluated by comparing DBSCAN-detected flares against the known injected events.
- **Model Training**: `Predictive_Models.ipynb` trains a predictive model using extracted flare-related features. The response variable is based on flares flagged by the DBSCAN algorithm. Model performance is assessed on a test set to evaluate its predictive accuracy.

## **Code Development and Contributions**

The code for this project was developed by **Diksha Sen Chaudhury**, under the supervision of **Dr. Vianey Leos Barajas**, course instructor for **STA2453** at the **University of Toronto**. This repository will be updated as new developments arise.
