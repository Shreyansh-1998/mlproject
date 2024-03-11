import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path=os.path.join('artifacts',"model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "RandomForestRegressor":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False)
            }

            params={

                "RandomForestRegressor":{'n_estimators':100,'max_depth':2},

                "DecisionTreeRegressor":{'max_depth':2,'min_samples_split':2},

                "LinearRegression":{},

                "KNeighborsRegressor":{'n_neighbors':2,'weights':'uniform'},

                "AdaBoostRegressor":{'n_estimators':100,'learning_rate':[0.01,0.1,1]},

                "GradientBoostingRegressor":{'n_estimators':100,'learning_rate':[0.01,0.1,1]},

                "XGBRegressor":{'n_estimators':100,'learning_rate':[0.01,0.1,1]},

                "CatBoostRegressor":{'n_estimators':100}
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            #To get the best model from the model report
            best_model_score=max(sorted(model_report.values()))

            #To get the best model name from dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(

                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )    

            predicted=best_model.predict(X_test)
            r2_score_value=r2_score(y_test,predicted)
            mse=mean_squared_error(y_test,predicted)
            rmse=np.sqrt(mse)

            return (
                f"Best_model_name: {best_model_name}",
                f"R2_score: {r2_score_value}"
                f" RMSE: {rmse}"
)

        except Exception as e:
            raise CustomException(e,sys)
