# Simple Tutorial
Hello! This code is about the higher-order link prediction problem, and the details of our work are presented in this paper "Higher-Order Link Prediction via Light Hypergraph Neural Network and Hybrid Aggregator".

## Installing Dependencies

First you need to install the dependency packages needed for this project. The following command can be run.
`pip install -r requirements.txt`

## Descriptions of documents

The "checkpoints" folder contains the model files that were temporarily stored during the training of the neural network, and we save the one with the highest accuracy on the validation set for testing.
The "datasets" folder contains the higher-order network data we use.
The "Aggregator" file contains all the aggregator structures we designed and tested.
The "config" file contains the configuration of all hyperparameters, which can be changed directly in this file.
The "HEBatch" file is used to batch the dataset.
The "LHGNN" file contains the specific structure of our improved hypergraph neural network.
The "run" file is the main file, which contains the main flow of the project.
The "train_test" file contains the specific operations for training and testing.
Finally, the "utils" file contains some utility functions.

## Run the Code

If the "config" file is modified you can simply run the "run" file directly, or you can specify the values of the parameters by the command.
`python run.py --dataset_name="email-Enron"...`

