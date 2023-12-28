import numpy as np
from pandas import DataFrame
from sklearn.utils._encode import _unique
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from ._base import SelectorMixin
from ..base import BaseEstimator


class WoeDirectionFilter(BaseEstimator, SelectorMixin):
    @_deprecate_positional_args
    def __init__(self, *, directions=None, regularization=1.0):
        self.directions = directions
        self.regularization = regularization

    def _validate_directions(self, X):
        _, n_features = X.shape
        directions = self.directions
        if directions is None:
            raise ValueError("direction is not None!")
        
        if isinstance(directions, str):
            if set(list(directions)) != {'+', '-'}:
                raise ValueError("{} received an invalid direction. Received {}, expected string composed of '+' and '-'.".format(self.__class__.__name__, directions))
            return np.full(n_features, directions, dtype=np.object)
        
        if isinstance(directions, dict):
            if isinstance(X, DataFrame):
                directions_ = []
                for f in X.columns:
                    # if f not in directions:
                    #     raise ValueError("No direction for feature {}!".format(f))
                    # direction = directions[f]
                    direction = directions.get(f, "++")
                    if not isinstance(direction, str):
                        raise ValueError("{) received an invalid direction type. Received {}, expected str.".format(self.__class__.__name__, type(direction).__name__))
                    if set(list(direction)) != {'+', '-'}:
                        raise ValueError("{} received an invalid direction. Received {}, expected string composed of '+' and '-'.".format(self.__class__.__name__, direction))
                    directions_.append(direction)
                directions_ = np.asarray(directions_, dtype=np.object)
            else:
                raise ValueError("Directions should be a dictionary only for dataframe input!")
        else:
            directions_ = check_array(directions, dtype=np.object, copy=True, ensure_2d=False)

            if directions_.ndim > 1 or directions_.shape[0] != n_features:
                raise ValueError("directions must be an array of string with shape (n_features,).")
            
            bad_directions_value = np.fromiter((set(list(x)) != {'+', '-'} for x in directions_), dtype=np.bool)

            violating_indices = np.where(bad_directions_value)[0]
            if violating_indices.shape[0] > 0:
                indices = ", ".join(str(i) for i in violating_indices)
                raise ValueError("{} received an invalid direction at indices {}. Direction must be a string composed of '+' and '-'.".format(self.__class__.__name__, indices))
        return directions_

    def _check_y(self, y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) != 2:
            raise ValueError("Only support binary label for bad rate encoding!")
        return y

    def fit(self, X, y=None, **fit_params):
        X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True)
        y = self._check_y(y)
        regularization = self.regularization
        bad_mask, good_mask = (y == 1), (y != 1)
        bad_tot, good_tot = np.count_nonzero(bad_mask) + 2 * regularization, np.count_nonzero(good_mask) + 2 * regularization
        residue = np.log(good_tot / bad_tot)
        _, n_features = X.shape
        support_mask = np.full(n_features, True, dtype=bool)

        directions = self._validate_directions(n_features)
        for i, direction in enumerate(directions):
            Xi = X[:, i]
            cats = _unique(Xi)
            woes = []
            append_woe = woes.append
            for cat in cats:
                mask = Xi == cat
                if np.count_nonzero(mask) == 1:
                    append_woe(0.)
                else:
                    bad_num = np.count_nonzero(np.logical_and(mask, bad_mask)) + regularization
                    good_num = np.count_nonzero(np.logical_and(mask, good_mask)) + regularization
                    append_woe(np.log(bad_num / good_num) + residue)

            # get woes' direction
            woe_directions = []
            last_trend = ''
            for i, w1, w2 in zip(range(len(woes)), woes[:-1], woes[1:]):
                trend = '+' if w2 - w1 >= 0 else '-'
                if i == 0:
                    woe_directions.append(trend)
                else:
                    if trend != last_trend:
                        woe_directions.append(trend)
                last_trend = trend

            woe_direction = ''.join(woe_directions)
            if direction == "++":
                continue
            if woe_direction != direction:
                support_mask[i] = False
        self.support_mask_= support_mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_

    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": False,
            "requires_y": True,
        }


def _gen_woe_value(x, y, regularization=1.0):
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization
    for cat in _unique(x):
        mask = x == cat
        # Ignore unique values. This helps to prevent overfitting on id-like columns.
        if np.count_nonzero(mask) == 1:
            yield 0.
        else:
            n_event = np.count_nonzero(np.logical_and(mask, event_mask)) + regularization
            n_nonevent = np.count_nonzero(np.logical_and(mask, nonevent_mask)) + regularization
            event_rate = n_event / event_tot
            nonevent_rate = n_nonevent / nonevent_tot
            woe = np.log(event_rate / nonevent_rate)
            yield woe


def _col_woe(x, y, regularization=1.0):
    return np.fromiter((w for w in _gen_woe_value(x, y, regularization=regularization)), dtype=np.float64)


def _col_woes(x, y, regularization=1.0):
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization
    uniques = _unique(x)
    n_cats = len(uniques)
    event_rates = np.zeros(n_cats, dtype=np.float64)
    nonevent_rates = np.zeros(n_cats, dtype=np.float64)
    for i, cat in enumerate(uniques):
        mask = x == cat
        event_rates[i] = np.count_nonzero(mask & event_mask) + regularization
        nonevent_rates[i] = np.count_nonzero(mask & nonevent_mask) + regularization

    # Ignore unique values. This helps to prevent overfitting on id-like columns.
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot
    woes = np.log(event_rates / nonevent_rates)
    woes[bad_pos] = 0.
    return woes
