"""Description: This file is the entry point of the application. 
It calls the DataIngestion, DataTransformation and ModelTrainer classes to perform the data ingestion, 
data transformation and model training respectively."""

import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


if __name__=="__main__":

    obj=DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))
        