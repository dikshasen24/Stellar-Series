# Stellar Flares Detection from Photometric Data Using Clustering and Machine Learning

This repository contains my project on detecting and predicting stellar flares from TESS photometric time-series data, with a focus on the star TIC 0131799991. An unsupervised clustering approach was used to detect flares, and a predictive model was trained to forecast future flare occurrences. The analysis includes preprocessing of flux data, feature engineering, flare injection and recovery simulations, and comparative evaluation of predictive models.


## **Code and Workflow**

- **Data-preprocessing and EDA**: `EDA.ipynb` handles missing value imputation using STL decomposition, and exploratory data analysis is conducted to examine underlying patterns in the light curve. 
- **Flare Detection (DBSCAN Clustering)**: `DBSCAN.ipynb` applies DBSCAN to detect flare candidates. Parameter tuning focuses on detecting both strong and weak flares.
- **Flare Injection & Recovery**: `Simulations.ipynb` simulates synthetic flares and injects them into two types of baselines: a low-noise synthetic baseline and a more realistic, data-aligned baseline. DBSCAN's performance is then evaluated using known injection times, with metrics such as sensitivity and precision used to assess detection effectiveness. 
- **Predictive Modeling**: `Predictive_Models.ipynb` trains models to predict future flare events using features derived from the flux time series. Flare labels are based on points identified by the DBSCAN clustering method. Both XGBoost and LSTM models are implemented and compared in terms of their ability to predict DBSCAN-detected flares. 
 
## **Code Development and Contributions**

The code for this project was developed by **Diksha Sen Chaudhury**, under the supervision of **Dr. Vianey Leos Barajas**, course instructor for **STA2453** at the **University of Toronto**. This repository will be updated as new developments arise.
