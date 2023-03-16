======================================
Predicting of Realty Prices 
======================================

Introduction
------------
This library implements ML model wich is used to predict prices 
of realty in Moscow Region. The model is trained on `Sberbank Russian Housing Market`_ dataset

Usage
-----

Train script
~~~~~~~~~~~~
Trains and saves the Model

Template
""""""""
::

    [~]$ python train.py <data_location> <save_path>

Input parameters

* <data_location> - location of train data (*.csv* file)
* <save_path> - location where should be saved trained model (saves as *model.json*)
  
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

    [~]$ python predict.py <data_location> <model_location>

Input parameters

* <data_location> - location of data to be predicted (*.csv* file)
* <save_path> - location of the trained model (*model.json* file)
  
Example
"""""""
::
    
    [~]$ python predict.py test.csv model.json
    +----------------+-------------------+
    | Transaction ID | Approximate Price |
    +----------------+-------------------+
    |     30474      |      5709956      |
    |     30475      |      8662689      |
    |     30476      |      5963731      |
    |     30477      |      6130314      |
    |     30478      |      5387550      |
    |     30479      |      9045907      |
    |     38135      |      9206236      |
    +----------------+-------------------+

Setup
-----
Check out setup guide `here`_

.. _`here`: #
.. _`Sberbank Russian Housing Market`: https://www.kaggle.com/competitions/sberbank-russian-housing-market/overview