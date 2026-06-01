"""Small compatibility guards for napari behavior used by this plugin."""

from __future__ import annotations

from types import MethodType

import numpy as np


def guard_empty_translucent_ray(layer) -> None:
    """Avoid napari status errors when a 3D translucent ray samples no voxels."""
    if getattr(layer, "_racc_empty_ray_guard", False):
        return
    if not hasattr(layer, "_calculate_value_from_ray"):
        return

    original = layer._calculate_value_from_ray

    def _calculate_value_from_ray(self, values):
        if np.size(values) == 0:
            return None
        return original(values)

    layer._calculate_value_from_ray = MethodType(_calculate_value_from_ray, layer)
    layer._racc_empty_ray_guard = True
