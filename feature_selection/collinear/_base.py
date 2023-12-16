from .._base import SelectorMixin


__all__ = ["CollinearSelectorMixin"]


# collinear selector is a kind of selector for collinear selection.
class CollinearSelectorMixin(SelectorMixin):
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
        }