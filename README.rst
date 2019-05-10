Neural Network based Sound Source Localization Models
=====================================================

This repository includes the programs to run and test the neural network models that we proposed in our publications in sound source localization and related fields, which are:
* Deep learning based direction-of-arrival (DOA) estimation
* Joint DOA estimation and speech/non-speech classification

The networks has been trained to work with the microphone array of the Softbank Pepper robot (early version with directional microphones).

Please cite the relevant publications when using the code.


Dependency
----------

* `Python <https://www.python.org/>`_ (version 2.7)
* `NumPy <http://www.numpy.org/>`_ (version 1.14)
* `PyTorch <https://pytorch.org/>`_ (version 0.2)
* `apkit <https://github.com/hwp/apkit>`_ (version 0.2)


Data
----

We use the `SSLR dataset <https://www.idiap.ch/dataset/sslr>`_ for the experiments.


Deep learning based direction-of-arrival (DOA) estimation
---------------------------------------------------------

This work is based on:

  Deep Neural Networks for Multiple Speaker Detection and Localization
  Weipeng He, Petr Motlicek, Jean-Marc Odobez 
  In *IEEE International Conference on Robotics and Automation (ICRA)*, 2018

The neural network models can simultaneously detect and localize multiple sound sources in noisy environment. In particular, we included the following items in the repository:

- code to extract features: GCC coefficients and GCCFB.
- trained neural network models: GCC-MLP, GCCFB-CNN, and GCCFB-TSNN.
- code to run the network models and save the output to files.
- scripts to evaluate and visualize results.

Usage
.....

(We are working on this)



Joint DOA estimation and speech/non-speech classification
---------------------------------------------------------

This work is based on:

  Joint Localization and Classification of Multiple Sound Sources Using a Multi-task Neural Network
  Weipeng He, Petr Motlicek, Jean-Marc Odobez 
  In *INTERSPEECH*, 2018

The neural network can detect and localize multiple sound sources, and classify them into speech or non-speech. In particular, we included the following items in the repository:

- trained neural network models: the multi-task NN as well as single-task NN for SSL only.
- code to run the network models and save the output to files.
- scripts to evaluate and visualize results.

Usage
.....

(We are working on this)



