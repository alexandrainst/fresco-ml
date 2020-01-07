# fresco-ml
[![Build Status](https://travis-ci.org/aicis/fresco-ml.svg?branch=master)](https://travis-ci.org/aicis/fresco-ml)

This repo contains various machine learning tools based on the secure multi-party framework FRESCO. These allows multiple parties to train and/or evaluate machine learning models on their combined data without sharing their data with the other parties.

The tools currently include:
* Neural networks
* Decision trees
* Federated learning
* Support vector machines
* Logistic regression

The easist way to see how each tool is used is to see the corresponding unit tests.

## Federated learning demo ##
The repo contains a demo of federated learning on the MNIST handwritten digits dataset where all parties run locally for 10 global epochs. Build it with `mvn package`and run `./launchscript.sh [number of parties] [number of local examples]`. Example parameters could be 3 parties and 5.000 local examples each.