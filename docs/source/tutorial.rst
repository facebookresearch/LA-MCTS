Tutorial
========

First we need a function to optimize. In this example, we use `Levy function <https://www.sfu.ca/~ssurjano/levy.html>`_.

.. code-block:: python

    class Levy(Func):
        def __init__(self, dims=100):
            self._dims = dims
            self._lb = -10 * np.ones(dims)
            self._ub = 10 * np.ones(dims)

        def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            w = 1 + (x - 1.0) / 4.0
            val = np.sin(np.pi * w[:, 0]) ** 2 + \
                  np.sum((w[:, 1:self._dims - 1] - 1) ** 2 *
                         (1 + 10 * np.sin(np.pi * w[:, 1:self._dims - 1] + 1) ** 2), axis=1) + \
                  (w[:, self._dims - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, self._dims - 1]) ** 2)
            return val, None

        def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                        classifier: ClassifierEnum = ClassifierEnum.THRESHOLD_SVM_CLASSIFIER) -> Dict:
            params = super().mcts_params(sampler, classifier)
            params["params"]["cp"] = 0.01
            params["params"]["cb_base"] = ConfidencyBase.Best
            params["params"]["leaf_size"] = 40
            params["params"]["num_samples_per_sampler"] = 100
            if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
                params["sampler"]["params"]["acquisition"] = "ei"
                params["sampler"]["params"]["nu"] = 2.5
            return params

Next we use :class:`lamcts.classifier.ThresholdSvmClassifier` and :class:`lamcts.sampler.RandomSampler` to optimize the function

.. code-block:: python

    def mcts_opt(call_budget: float = 1000):
        func = Levy()
        func_wrapper = StatsFuncWrapper(func)
        params = func.mcts_params(SamplerEnum.RANDOM_SAMPLER, ClassifierEnum.THRESHOLD_SVM_CLASSIFIER)
        mcts = MCTS.create_mcts(func_wrapper, func_wrapper, params)
        st = time.time()
        try:
            mcts.search(greedy=GreedyType.ConfidencyBound, call_budget=call_budget)
        except TimeoutError:
            pass
        wt = time.time() - st
        return func_wrapper.stats, wt

Above function returns calling history result and wall time. To get the results, simply do:

.. code-block:: python

    stats, wt = mcts_opt()

    if len(stats.call_history) > 0:
        cp = stats.call_history[-1]
        print(f"best: {cp.fx}, call mark: {cp.call_mark}, time mark: {cp.time_mark}")
    print(f"total calls: {stats.total_calls}, wall time: {wt}")
