
# Capstone Project - Azure Machine Learning Engineer

The aim of the project is to create two models, compare their performance, deploy the best performing model as a webservice and test the model's endpoint. The first model was trained using Automated ML and the second one  was created by  hyperparameter tuning using HyperDrive. 

## Dataset

### Overview
For this project, I am using  the Heart Failure Prediction dataset from Kaggle. The dataset contains 12 clinical features of 299 patients with heart failure and a target variable "DEATH EVENT" indicating if the patient deceased during the follow-up period (boolean). Machine Learning can help detect and manage high-risk patients at an earlier stage.

### Task
The task is to detect high-risk patients. To solve this binary classification problem, I will use the 12 features to predict possible death events with the help of an ML algorithm. 
12 clinical features: age, anaemia, diabetes, creatinine_phosphokinase, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, and time. Target variable: DEATH_EVENT
Clinical Features | Description 
------------- | ----------- 
Age | Age of the patient
anaemia  | Decrease of red blood cells or hemoglobin (boolean)
creatinine_phosphokinase | Level of the CPK enzyme in the blood (mcg/L)
diabetes | If the patient has diabetes (boolean)
ejection_fraction | Percentage of blood leaving the heart at each contraction (percentage)
high_blood_pressure | If the patient has hypertension (boolean)
platelets | Platelets in the blood (kiloplatelets/mL)
serum_creatinine | Level of serum creatinine in the blood (mg/dL)
serum_sodium | Level of serum sodium in the blood (mEq/L)
sex | Woman or man (binary)
smoking | If the patient smokes or not (boolean)
time | Follow-up period (days)

Target Variable | Description 
------------- | ----------- 
DEATH_EVENT | If the patient deceased during the follow-up period (boolean)



### Access
In the AutoML part of the project, I  imported the dataset after I had registered it in Azure Workspace.   

'''found = False
key = "Heart Failure Prediction"
description_text = "Heart Failure Prediction DataSet for Udacity Project 3"
if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key]'''

In the Hyperdrive part of the project, I saved the dataset to my github repository and retrieved the data from the URL using TabularDatasetFactory class in train.py script. 


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
