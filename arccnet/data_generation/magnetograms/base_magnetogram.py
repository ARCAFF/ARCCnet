from abc import ABC

import drms

import arccnet.data_generation.utils.default_variables as dv

__all__ = ["BaseMagnetogram"]


class BaseMagnetogram(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._c = drms.Client(debug=False, verbose=False, email=dv.JSOC_DEFAULT_EMAIL)
