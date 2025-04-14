# Stellar Flares Detection from Photometric Data Using Clustering and Machine Learning

This repository contains our work on using **unsupervised clustering** and **predictive modeling** to detect stellar flares from **TESS photometric time-series data**. Our goal is to develop an approach that can accurately identify stellar flares and predict future occurrences. We preprocess flux data from **TIC 0131799991**, perform exploratory data analysis (EDA), and apply clustering techniques for initial flare detection. A predictive model is then trained using the detected flare data.

## **Code and Workflow**

- **Data-preprocessing and EDA**: The script `EDA.ipynb` imputes missing values into the time series and performs exploratory data analysis (EDA) to understand the data. 
- **Flare Detection**: The script `DBSCAN.ipynb` applies **DBSCAN clustering** to identify potential stellar flares.
- **Flare Injection & Recovery**: Flares are simulated using `Simulations.ipynb`, allowing for controlled injection of synthetic flares into the light curve. The detection accuracy of the DBSCAN algorithm will be assessed by comparing detected vs. injected flares.
- **Model Training**: The script `Predictive_Models.ipynb` will train a predictive model using extracted flare features. Model performance will be evaluated on a test set to determine accuracy in flare prediction.

## **Code Development and Contributions**

The code for this project was developed by **Diksha Sen Chaudhury**, under the supervision of **Dr. Vianey Leos Barajas**, course instructor for **STA2453** at the **University of Toronto**. This repository will be updated as new developments arise.
