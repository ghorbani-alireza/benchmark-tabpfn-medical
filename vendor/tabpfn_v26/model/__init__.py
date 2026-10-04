"""Contains references to the base architecture, for backwards compatability.

DEPRECATED: import tabpfn_v26.architectures.base instead

Previously tabpfn only supported a single architecture, which was in this tabpfn.model
module. Now we support multiple architectures, stored in tabpfn.architectures, and
tabpfn.model has moved to tabpfn.architectures.base .
"""

import warnings

from tabpfn_v26 import model_loading as loading
from tabpfn_v26.architectures import encoders
from tabpfn_v26.architectures.base import (
    attention,
    bar_distribution,
    config,
    layer,
    memory,
    mlp,
    transformer,
)

__all__ = [
    "attention",
    "bar_distribution",
    "config",
    "encoders",
    "layer",
    "loading",
    "memory",
    "mlp",
    "transformer",
]

warnings.warn(
    "tabpfn.model has moved to tabpfn.architectures.base. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
