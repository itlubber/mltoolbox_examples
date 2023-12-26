from ._base import Pipeline, _name_estimators
from ..postprocessing import FeaturesOutPut


class FeaturesOutputPipeline(Pipeline):
    def _features_output(self, X):
        Xt = X
        features_output = {}
        for _, name, transform in self._iter():
            if isinstance(transform, FeaturesOutputPipeline):
                Xt, partial_output = transform._features_output(Xt)
                features_output[name] = partial_output
            elif isinstance(transform, FeaturesOutPut):
                output = transform.features_output(Xt)
                features_output[name] = output
                continue
            else:
                Xt = transform.transform(Xt)
        return Xt, features_output
    
    def features_output(self, X):
        _, features_output = self._features_output(X)
        return features_output


def make_features_out_pipeline(*steps, memory=None, verbose=False):
    return FeaturesOutputPipeline(_name_estimators(steps), memory=memory, verbose=verbose)
