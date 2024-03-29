Neural Network based Sound Source Localization Models
=====================================================

This repository includes the programs to run and test the neural network models that we proposed in our publications on sound source localization, including

* Deep learning based direction-of-arrival (DOA) estimation
* Joint DOA estimation and speech/non-speech classification

The networks has been trained to work with the microphone array of the Softbank Pepper robot (early version with directional microphones).

Please cite the relevant publications when using the code.


Dependency
----------

* `Python <https://www.python.org/>`_ (version >= 3.8)
* `NumPy <http://www.numpy.org/>`_ (version >= 1.14)
* `PyTorch <https://pytorch.org/>`_ (version >= 0.2)
* `apkit <https://github.com/hwp/apkit>`_ (version >= 0.2)


Data
----

We use the `SSLR dataset <https://www.idiap.ch/dataset/sslr>`_ for the experiments.


Usage
-----

Training
********

This command reproduces the "ResNet-STFT" model described in `"Deep Learning Approaches for Auditory Perception in Robotics", Weipeng He, PhD Thesis, EPFL <https://infoscience.epfl.ch/record/283940>`_:

.. code::

  ssl_nn_v2/train_resnet_2stage.py --model=models/test --window-size=8192 --hop-size=4096 --n-doa=360 --output-act=5 --n-out-hidden=1 --sigma=8 --s1-epoch=4 --epoch=10 --lr=0.001 --ld=2 --batch-size=128 <SSLR_PATH>/lsp_train_*

Inference
*********

.. code::

  ssl_nn_v2/test_nn_raw.py --feature=stft --method=METHOD_NAME --window-size=8192 --hop-size=4096 --batch-size=100 <SSLR_PATH>/human models/thesis_resnet_act5_p1lsp_s1ep4_ep10_valid_b100

Evaluation
**********

.. code::

  eval/gen_2tasks_report.py --method=METHOD_NAME --window-size=8192 --hop-size=4096 --output=REPORT_DIR --ssl-only <SSLR_PATH>/human
  gnuplot --persist REPORT_DIR/ssl_pr_plot

Publications
------------

The models and code in this repository are based on the work published in:

  Deep Neural Networks for Multiple Speaker Detection and Localization
  Weipeng He, Petr Motlicek, Jean-Marc Odobez 
  In *IEEE International Conference on Robotics and Automation (ICRA)*, 2018

  Joint Localization and Classification of Multiple Sound Sources Using a Multi-task Neural Network
  Weipeng He, Petr Motlicek, Jean-Marc Odobez 
  In *INTERSPEECH*, 2018

  Deep Learning Approaches for Auditory Perception in Robotics
  Weipeng He
  PhD Thesis, EPFL


