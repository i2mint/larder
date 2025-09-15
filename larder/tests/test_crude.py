"""
Test prepare_for_crude_dispatch for stored.crude
"""

from larder.crude import prepare_for_crude_dispatch
from functools import partial


def apply_model(fitted_model, fvs, method="predict"):
    method_func = getattr(fitted_model, method)
    return method_func(list(fvs))


def learn_model(learner, fvs, method="fit"):
    method_func = getattr(learner, method)
    return method_func(list(fvs))


class TinyMinMaxModel:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X_max = max(X)
        self.X_min = min(X)
        return self

    def predict(self, X):
        def min_max(x):
            return max(min(x, self.X_max), self.X_min)

        return [min_max(x) for x in X]


mall = dict(
    learner=dict(TinyMinMaxModel=TinyMinMaxModel()),
    fvs=dict(train_fvs_1=[[1], [2], [3], [5], [4], [2], [1], [4], [3]]),
    fitted_model=dict(
        fitted_model_1=TinyMinMaxModel().fit(
            [[1], [2], [3], [5], [4], [2], [1], [4], [3]]
        )
    ),
    model_results=dict(),
)


def test_prepare_for_crude_dispatch():
    dispatchable_learn_model = prepare_for_crude_dispatch(
        learn_model,
        param_to_mall_map=dict(learner="learner", fvs="fvs"),
        mall=mall,
        output_store="fitted_model",
    )

    dispatchable_apply_model = prepare_for_crude_dispatch(
        apply_model,
        param_to_mall_map=dict(fitted_model="fitted_model", fvs="fvs"),
        mall=mall,
        output_store="model_results",
        save_name_param="save_name_for_apply_model",
    )

    dispatchable_learn_model = partial(
        dispatchable_learn_model, learner="TinyMinMaxModel", fvs="train_fvs_1"
    )
    dispatchable_apply_model = partial(
        dispatchable_apply_model, fitted_model="fitted_model_1", fvs="train_fvs_1"
    )

    test_fvs = [[-100], [1], [2], [10]]
    tmm = dispatchable_learn_model()
    assert tmm.predict(test_fvs) == [[1], [1], [2], [5]]

    am = dispatchable_apply_model(save_name_for_apply_model="apply_model_results")
    assert am == [[1], [2], [3], [5], [4], [2], [1], [4], [3]]
    assert list(mall["model_results"]) == ["apply_model_results"]
    assert mall["model_results"]["apply_model_results"] == [
        [1],
        [2],
        [3],
        [5],
        [4],
        [2],
        [1],
        [4],
        [3],
    ]
