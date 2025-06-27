import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()

    def data_transformation(self):
        try:
            logging.info("Starting data transformation process")

            # Define the preprocessing steps
            numerical_features = ['reading_score', 'writing_score']
            categorical_features =  [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
                ]
            
            numerical_transformer = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))  # with_mean=False for sparse matrices
                ]
            )

            logging.info("Numerical and categorical transformers defined")
            logging.info("Numerical features: %s", numerical_features)
            logging.info("Categorical features: %s", categorical_features)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            logging.info("Preprocessing steps defined")

            return preprocessor
        
        except Exception as e:
            logging.error("An error occurred in the data transformation component.")
            raise CustomException(e, sys) from e
        

        def initiate_data_transformation(self, train_path, test_path):
            try:
                logging.info("Starting data transformation process")

                # Load the train and test datasets
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Train and test datasets loaded successfully")

                # Define the preprocessing steps
                preprocessor = self.data_transformation()

                numerical_features = ['reading_score', 'writing_score']
                target_column = 'math_score'

                input_features_train = train_df.drop(columns=[target_column], axis=1)
                target_feature_train = train_df[target_column]

                input_features_test = test_df.drop(columns=[target_column], axis=1)
                target_feature_test = test_df[target_column]

                logging.info("Input features and target feature separated")

                input_train_array = preprocessor.fit_transform(input_features_train)
                input_test_array = preprocessor.transform(input_features_test)

                train_array = np.c_[input_train_array, np.array(target_feature_train)]
                test_array = np.c_[input_test_array, np.array(target_feature_test)]

                logging.info("Data transformation completed successfully")

                # Save the preprocessor object
                save_object(
                    preprocessor = preprocessor,
                    file_path =  self.preprocessor_config.preprocessor_obj_file_path
                )

                return (
                    train_array,
                    test_array,
                    self.preprocessor_config.preprocessor_obj_file_path
                )

            except Exception as e:
                logging.error("An error occurred in the data transformation component.")
                raise CustomException(e, sys) from e