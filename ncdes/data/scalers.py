import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
from sklearn.base import TransformerMixin


class TrickScaler(TransformerMixin):
    """Tricks an sklearn scaler so that it uses the correct dimensions.

    Performs normal scaling on 3D tensors. If the shape is [N, L, C] where we wish to normalise by channel, this
    function converts the tensor onto shape [N * L, C], scales the columns according to the specified method and then
    converts back onto shape [N, L, C].
    """
    def __init__(self, scaling):
        # Setup scaling
        self.scaling = scaling
        if scaling == 'stdsc':
            scaler = StandardScaler()
        elif scaling == 'ma':
            scaler = MaxAbsScaler()
        elif scaling == 'mms':
            scaler = MinMaxScaler()
        elif (scaling is None) or (scaling is False):
            scaler = FunctionTransformer(func=None)
        else:
            raise NotImplementedError('Not implemented scaling method {}'.format(scaling))

        self.scaler = scaler

    def _trick(self, X):
        return X.reshape(-1, X.shape[2])

    def _untrick(self, X, shape):
        return X.reshape(shape)

    def fit(self, X, y=None):
        self.scaler.fit(self._trick(X), y)
        return self

    def transform(self, X):
        X_tfm = self.scaler.transform(self._trick(X))
        return torch.Tensor(self._untrick(X_tfm, X.shape))


