======================================
Predicting of Realty Prices 
======================================

Introduction
------------
This library implements ML model which is used to predict prices 
of realty in Moscow Region. The model is trained on `Sberbank Russian Housing Market`_ dataset

Usage
-----

Train script
~~~~~~~~~~~~
Trains and saves the Model

Template
""""""""
::

    [~]$ python train.py <data_path> <save_path>

Input parameters

positional arguments:
  * data_path - Path of train data
  * save_path - Path to save trained model

options:
  * -h, --help - show this help message and exit
  * --objective OBJECTIVE - Train objective
  * --n_estimators N_ESTIMATORS - Number of gradient boosted trees
  * --max_depth MAX_DEPTH - Maximum tree depth for base learners
  * --eta ETA - Boosting learning rate
  * --subsample SUBSAMPLE - Subsample ratio of the training instance
  * --colsample_bytree COLSAMPLE_BYTREE - Subsample ratio of columns when constructing each tree
  * --reg_lambda REG_LAMBDA - L2 regularization term on weights
  * --random_state RANDOM_STATE - Random number seed
  * --early_stopping_rounds EARLY_STOPPING_ROUNDS - Activates early stopping
  
Example
"""""""
::
    
    [~]$ ls
    [~]$ python train.py train.csv .
    [~]$ ls
    model.json

Predict script
~~~~~~~~~~~~~~
Predicts the prices and outputs it

Template
""""""""
::

    [~]$ python predict.py <data_path> <model_path> <save_path>

Input parameters

positional arguments:
  * data_path - Path of data to predict
  * model_path - Path of trained model
  * save_path - Path of saved predictions
  
Example
"""""""
::
    
    [~]$ python predict.py <data_path> <model_path> <save_path>
    [~]$ ls <save_path>
    predictions.npy

Setup
-----
Check out setup guide `here`_

.. _`here`: #
.. _`Sberbank Russian Housing Market`: https://www.kaggle.com/competitions/sberbank-russian-housing-market/overview
