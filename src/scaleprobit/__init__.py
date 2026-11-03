"""scaleprobit: threshold-crossing probabilities with a learned scale, and transfer of that scale
from a large source sample to a small target sample through a source-centred ridge.

    from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV, fit_source
"""
from ._model import ScaleProbitClassifier
from ._prior import Prior
from .transfer import DEFAULT_LAMS, ScaleProbitTransferCV, fit_source

__version__ = "0.1.0"

__all__ = ["DEFAULT_LAMS", "Prior", "ScaleProbitClassifier", "ScaleProbitTransferCV", "fit_source", "__version__"]
