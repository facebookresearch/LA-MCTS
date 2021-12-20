Usage
=====

Installation
------------

.. code-block:: console

   $ python setup.py install

API References
--------------

Function Interfaces
*******************

.. autoclass:: lamcts.Func
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.CheckPoint
    :members:

.. autoclass:: lamcts.CallHistory

.. autoclass:: lamcts.FuncStats
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.StatsFuncWrapper
    :members:
    :undoc-members:
    :show-inheritance:

MCTS
******************

.. autoclass:: lamcts.MCTS
    :members:
    :undoc-members:
    :show-inheritance:

Classifiers
***********

.. autoclass:: lamcts.classifier.Classifier
    :members:

.. autoclass:: lamcts.classifier.SvmClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.classifier.KmeanSvmClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.classifier.ThresholdSvmClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.classifier.RegressionSvmClassifier
    :members:
    :undoc-members:
    :show-inheritance:

Samplers
********

.. autoclass:: lamcts.sampler.Sampler
    :members:
    :show-inheritance:

.. autoclass:: lamcts.sampler.RandomSampler
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.sampler.BOSampler
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.sampler.TuRBOSampler
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.sampler.CmaesSampler
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.sampler.NevergradSampler
    :members:
    :undoc-members:
    :show-inheritance:

Utilies
*******

.. autoclass:: lamcts.Sample
    :members:

.. autoclass:: lamcts.Bag
    :members:
    :undoc-members:
    :show-inheritance:

Configuration
*************

.. autoclass:: lamcts.config.SearchType
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.config.SamplerEnum
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: lamcts.config.ClassifierEnum
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: lamcts.config.get_mcts_params