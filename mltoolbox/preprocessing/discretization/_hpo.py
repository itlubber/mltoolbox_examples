import inspect
import numbers
import numpy as np
import optuna
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import _deprecate_positional_args
from ._base import BaseDiscretizer
from ...base import MetaEstimatorMixin
from ...feature_selection._iv_selector import _col_iv


class _OptunaTrialBinEdges:
    _attribute_template = "i{}"

    def __init__(self, pre_bin_edges, n_bin, trial=None):
        self.pre_bin_edges = pre_bin_edges
        n_pre_bin = len(pre_bin_edges)
        self.n_pre_bin = n_pre_bin
        self.n_bin = n_bin
        self.trial = trial
    
    def __len__(self):
        return self.n_bin + 1

    def __iter__(self):
        template = self._attribute_template
        n_bin = self.n_bin
        trial = self.trial
        n_pre_bin = self.n_pre_bin
        indices = []
        for i in range(n_bin + 1):
            name = template.format(i)
            if i == 0:
                idx = 0
            elif i == n_bin:
                idx = n_pre_bin - 1
            else:
                idx = trial.suggest_int(name, min(max(indices) + 1, n_pre_bin - 2), n_pre_bin - 2)
            setattr(self, name, idx)
            indices.append(idx)
            yield idx

    def __getitem__(self, item):
        return self.values[item]

    def __eq__(self, other):
        if not isinstance(other, _OptunaTrialBinEdges):
            raise TypeError
        v1, v2 = self.values, other.values
        if len(v1) != len(v2):
            return False
        return np.all(v1 == v2)

    @property
    def values(self):
        return self.pre_bin_edges[np.fromiter((x for x in self.__iter__()), dtype=int)]


class OptunaDiscretizer(BaseDiscretizer, MetaEstimatorMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_bins=5, n_jobs=None, objective=_col_iv, direction="maximize", n_trials=100, trial_timeout=None, verbose=0):
        super(OptunaDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)
        self.estimator = estimator
        self.objective = objective
        self.direction = direction
        self.n_trials = n_trials
        self.trial_timeout = trial_timeout
        self.verbose = verbose

    def _validate_params(self):
        if not isinstance(self.estimator, BaseDiscretizer):
            raise ValueError("estimator should be of type BaseDiscretizer.")
        
        objective = self.objective
        if not callable(objective):
            raise ValueError()
        arg_spec = inspect.getfullargspec(objective)
        if arg_spec.args[:2] != ['x', 'y']:
            raise ValueError()
        
        direction = self.direction
        if direction not in ("minimize", "maximize"):
            raise ValueError()
        
        n_trials = self.n_trials
        if not isinstance(n_trials, numbers.Integral):
            raise ValueError()
        if n_trials <= 0:
            raise ValueError()
    
    def fit(self, X, y=None, **fit_params):
        self._validate_params()
        self.estimator.fit(X, y=y, **fit_params)
        return super (OptunaDiscretizer, self).fit(X, y=y)
    
    @property
    def closed(self):
        return self.estimator.closed
    
    @closed.setter
    def closed(self, value):
        self.estimator.closed = value
    
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        pre_bin_edges = self.estimator.bin_edges_[i]
        n_pre_bins = len(pre_bin_edges) - 1
        n_bin = min(n_pre_bins, n_bin)

        right = self.closed == 'right'
        objective = self.objective
    
        def func(trial):
            trial_bin_edges = _OptunaTrialBinEdges(pre_bin_edges, n_bin, trial=trial).values
            xt = np.digitize(x, trial_bin_edges[1:], right=right)
            np.clip(xt, 0, n_bin - 1, out=xt)
            return objective(xt, y)

        study = optuna.create_study(study_name="Auto binning feature #{)".format(i), direction=self.direction)
        study.optimize(func, n_trials=self.n_trials, timeout=self.trial_timeout, n_jobs=1 if self.n_jobs is None else self.n_jobs, show_progress_bar=bool(self.verbose))
        best_params = study.best_params
        template = _OptunaTrialBinEdges._attribute_template
        indices = np.asarray([0, *(best_params[template.format(i)] for i in range(1, n_bin)), n_pre_bins - 1], dtype=int)
        bin_edge = pre_bin_edges[indices]
        return bin_edge, len(bin_edge) - 1
    
    def _more_tags(self):
        return {
            "X_types": _safe_tags(self.estimator, "X_types"),
            "allow_nan": _safe_tags(self.estimator, "allow_nan"),
            "requires_y": _safe_tags(self.estimator, "requires_y"),
        }


class Discretizationobjective:
    def __call__(self, x, y, **kwargs):
        pass
