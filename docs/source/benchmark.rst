Benchmark
=========

How to Run Benchmarks
---------------------

.. code-block:: console

   $ cd example
   $ python run_benchmark.py -f levy100 -s mcts -m turbo -l threshold_svm -r 40 -t 180

In above example, *Levy100* is optimized with LA-MCTS as optimizer. TuRBO sampler and threhold SVM classifier are used.
Total 40 runs are executed, each with 180 second time budget, the results will be saved in *example/output* folder.

The results can then be reported by

.. code-block:: console

   $ python benchmark_result.py -f levy100

Only top 10 best runs are used for the report.

Test Functions
--------------

.. image:: images/levy100.png
  :width: 800
  :alt: Levy100 results

.. image:: images/ackley100.png
  :width: 800
  :alt: Ackley100 results

.. image:: images/rastrigin100.png
  :width: 800
  :alt: Rastrigin100 results

Control Functions
-----------------

.. image:: images/push.png
  :width: 800
  :alt: Push results

.. image:: images/lunarlanding.png
  :width: 800
  :alt: Lunar Landing results

Mujuco Policies
---------------

.. image:: images/hopper.png
  :width: 800
  :alt: Hopper results

.. image:: images/humanoid.png
  :width: 800
  :alt: Humanoid results

.. image:: images/walker_2d.png
  :width: 800
  :alt: Walker 2D results
