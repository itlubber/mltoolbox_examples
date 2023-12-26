import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, column_or_1d
from sklearn.utils._encode import _unique
from sklearn.utils.validation import FLOAT_DTYPES
from ._base import _ATOL, _RTOL
from ._utils import get_bin_repr


def get_binning_table(i, x, y, bin_edge, *, closed='left', precision=2, regularization=1.0, add_summary=True):
    x = column_or_1d(x)
    y = column_or_1d(y)

    n_bins = len(bin_edge) - 1
    x = check_array(x, dtype=FLOAT_DTYPES, force_all_finite=True, ensure_2d=False, ensure_min_samples=2 * n_bins)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Only support binary label for computing binning table!")
    
    eps = _ATOL + _RTOL * np.abs(x)
    xt = np.digitize(x + eps, bin_edge[1:], right=(closed == 'right'))
    np.clip(xt, 0, n_bins - 1, out=xt)

    n_samples = len(x)
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask)
    nonevent_tot = np.count_nonzero(nonevent_mask)

    bin_reprs = []
    bin_sizes = []
    bin_rates = []
    n_events = []
    n_nonevents = []
    bin_event_rates = []
    bin_nonevent_rates = []
    woes = []
    ivs = []

    bin_indices = _unique(xt).tolist()

    for idx, l, r in zip(bin_indices, bin_edge[:-1], bin_edge[1:]):
        bin_repr = get_bin_repr(idx, l, r, closed=closed, precision=precision)
        bin_reprs.append(bin_repr)

        mask = xt == idx
        bin_size = np.count_nonzero(mask)
        bin_sizes.append(bin_size)
        bin_rates.append(bin_size / n_samples)

        n_event = np.count_nonzero(mask & event_mask)
        n_nonevent = np.count_nonzero(mask & nonevent_mask)
        n_events.append(n_event)
        n_nonevents.append(n_nonevent)
        bin_event_rates.append(n_event / bin_size)
        bin_nonevent_rates.append(n_nonevent / bin_size)

        if np.count_nonzero(mask) == 1:
            woe = 0.
            iv = 0.
        else:
            event_rate = (n_event + regularization) / (event_tot + 2 * regularization)
            nonevent_rate = (n_nonevent + regularization) / (nonevent_tot + 2 * regularization)
            woe = np.log(event_rate / nonevent_rate)
            iv = (event_rate - nonevent_rate) * woe
        woes.append(woe)
        ivs.append(iv)

    if add_summary:
        bin_indices.append(f"Summary of feature {i}")
        bin_reprs.append("(-inf, inf)")
        bin_sizes.append(n_samples)
        bin_rates.append(1.)
        n_events.append(event_tot)
        n_nonevents.append(nonevent_tot)
        bin_event_rates.append(event_tot / n_samples)
        bin_nonevent_rates.append(nonevent_tot / n_samples)
        woes.append(sum(woes))
        ivs.append(sum(ivs))
    data = {
        "Bin": bin_reprs,
        "Bin size": bin_sizes,
        "Sample rate": bin_rates,
        "Event": n_events,
        "Non-event": n_nonevents,
        "Event rate": bin_event_rates,
        "Non-event rate": bin_nonevent_rates,
        "WoE": woes,
        "IV": ivs,
    }
    binning_table = pd.DataFrame.from_dict(data=data)
    binning_table.index = bin_indices
    return binning_table
