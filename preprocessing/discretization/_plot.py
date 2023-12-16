import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, column_or_1d
from sklearn.utils._encode import _unique
from sklearn.utils.validation import FLOAT_DTYPES, check_consistent_length
from ._base import _ATOL, _RTOL
from ._utils import get_bin_repr


def p1ot_binning_bivar(i, x, y, bin_edge, *, closed='left', precision=2, regularization=1.0):
    """Plot binning bivar

    Parameters
    ----------
    i : int
        Feature index, also used in figure title.
    x: array-like of shape (n_samples,)
        Feature value before binning
    y : array-like of shape (n_samples,)
        Label value
    bin_edge : array-like of shape (n_bins + 1,)
        Binning edges
    closed : str, default='left', can be 'right'
        Binning edges' close side.
    precision : int, default=2
        Float point format precision.
    regularization : float, default=1.e
        WoE regularization
    """
    check_consistent_length(x, y)
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

    bin_indices = _unique(xt)

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

        if bin_size == 1:
            woe = 0
            iv = 0
        else:
            event_rate = (n_event + regularization) / (event_tot + 2 * regularization)
            nonevent_rate = (n_nonevent + regularization) / (nonevent_tot + 2 * regularization)
            woe = np.log(event_rate / nonevent_rate)
            iv = (event_rate - nonevent_rate) * woe
        woes.append(woe)
        ivs.append(iv)

    fig, ax1 = plt.subplots()

    p2 = ax1.bar (range(n_bins), n_events, color="tab:red")
    p1 = ax1.bar(range(n_bins), n_nonevents, color="tab:blue", bottom=n_events)

    handles = [p1[0], p2[0]]
    labels = [ "Non-event", "Event"]

    ax1.set_xlabel("Bin ID", fontsize=10)
    ax1.set_ylabel("Bin size", fontsize=10)
    plt.xticks(range(n_bins), bin_reprs, rotation=25, fontsize=6)

    ax2 = ax1.twinx()

    ax2.plot(range(n_bins), woes, linestyle="solid", marker="o", color="black")
    ax2.set_ylabel("WoE", fontsize=10)
    ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

    plt.title("Feature {} Bi-var (IV = {})".format(i, format(sum(ivs), f".{precision}f")), fontsize=12)
    plt.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
