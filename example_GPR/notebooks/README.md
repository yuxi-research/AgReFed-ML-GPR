# Machine Learning with Gaussian Process Regression

This workflow shows an example of using machine learning methods with Gaussian Process Regression (GPR) to predict the spatio-temporal soil moisture in top surface and root zone. The workflow is based on the GitHub repository [https://github.com/Sydney-Informatics-Hub/AgReFed-ML]. All requirements algin with the repository.

## Data
This dataset covers 31 soil moisture probe stations from 2019-12-01 to 2021-12-31 (2 years and 1 month) in CSIRO's Borrowa Agricultural Research Station (BARS). The dataset includes the following variables:
- DepthTop: depth of top surface (cm)
- DepthBot: depth of root zone (cm)
- Rain: daily rainfall (mm)
- Longitude: longitude of the site
- Latitude: latitude of the site
- DEM: digital elevation model (m)
- Slope: slope of the site
- TWI: topographic wetness index
- K: Radiometric that estimates concentration of potasium (K)
- T: Radiometric that estimates concentration of thorium (T)
- U: Radiometric that estimates concentration of uranium (U)
- Total: Radiometric that estimates concentration of total (K + T + U)
- Solar: solar radiation (MJ/m2)
- NDVI_05: 5 percentile NDVI of historical data
- NDVI_50: 50 percentile NDVI of historical data
- NDVI_95: 95 percentile NDVI of historical data
- Rain_df_50: lagged rainfall with discount factor of 0.5
- Rain_df_70: lagged rainfall with discount factor of 0.7
- Rain_df_90: lagged rainfall with discount factor of 0.9
- Rain_df_95: lagged rainfall with discount factor of 0.95
- Rain_df_99: lagged rainfall with discount factor of 0.99
- Rain_df_999: lagged rainfall with discount factor of 0.999
- ET20m_df: Evapotranspiration (mm) at 20 m resolution.
- ET20m_df_50: lagged ET20m with discount factor of 0.5
- ET20m_df_70: lagged ET20m with discount factor of 0.7
- ET20m_df_90: lagged ET20m with discount factor of 0.9
- ET20m_df_95: lagged ET20m with discount factor of 0.95
- ET20m_df_99: lagged ET20m with discount factor of 0.99
- ET20m_df_999: lagged ET20m with discount factor of 0.999
- Bucket: bucket size (mm)
- CLY: clay content (%)
- SND: sand content (%)
- SLT: silt content (%)
- SOC: soil organic carbon (%)
- SM: volumetric soil moisture (%)

Daily: dataset.csv
Weekly: datset_weekly.csv

## Machine learning model options
- rf: Random Forest
- rf-gp: Random Forest as mean function of Gaussian Process Regression
- blr: Bayesian Linear Regression
- blr-gp: Bayesian Linear Regression as mean function of Gaussian Process Regression
- const: Constant model (mean of training data)
- gp-only: Constant model as mean function of Gaussian Process Regression


## Workflow
This workflow follows the Spatial-Temporal Model of the Agrefed respotory. The workflow is based on the following steps:
1. Feature selection: feature_selection_moisture.ipynb (this notebook includes functions for generating settings file)
2. Model testing:  
    Testing temporal prediction: testmodels_temporal.ipynb, configure settings in settings_testmodel_temporal.yaml.  
    Testing spatial prediction: testmodels_spatial.ipynb, configure settings in settings_testmodel_spatial.yaml.
3. Model prediction:  
    Predicting temporal prediction: predict_temporal.ipynb, configure settings in settings_predict_temporal.yaml.  
    Predicting spatial prediction: predict_spatial.ipynb, configure settings in settings_predict_spatial.yaml.
