import json
import numpy as np
from pandas import DataFrame
from py4j.java_gateway import JavaGateway

from ._feature_derive import NumExprDerive


PYTHON_PROXY_PORT = 25333
java_gateway = JavaGateway(python_proxy_port=PYTHON_PROXY_PORT)


class AviExprDerive(NumExprDerive):
    def __init__(self, derivings):
        super(AviExprDerive, self).__init__(derivings)
        gateway = java_gateway
        self.app = gateway.entry_point
    
    def _transform_frame(self, X):
        feature_names = X.columns.tolist()
        self.features_names = feature_names
        index = X.index
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        derived_names = []
        for i, (name, expr) in enumerate(self.derivings):
            derived_names.append(name)
        data = self.app.transform(json.dumps(X.tolist()), json.dumps(self.derivings), json.dumps(self.features_names))
        columns = feature_names + derived_names
        return DataFrame(data=data, columns=columns, index=index)
        
    def _transform_ndarray(self, X):
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        data = self.app.transform(json.dumps(X.tolist()), json.dumps(self.derivings), json.dumps([str(i) for i in range(len(X))]))
        return np.array(data)
