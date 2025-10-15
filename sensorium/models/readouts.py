from neuralpredictors.layers.readouts import (FullGaussian2d,
                                              MultiReadoutSharedParametersBase)


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d
