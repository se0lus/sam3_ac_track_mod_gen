"""
SAM3 action modules package.

Each action module should expose:
  - ACTION_SPECS: list[ActionSpec]
or:
  - def get_action_specs() -> list[ActionSpec]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Type

import bpy  # type: ignore[import-not-found]


@dataclass(frozen=True)
class ActionSpec:
    """
    A single menu action backed by a Blender operator.

    - operator_cls: bpy.types.Operator subclass
    - menu_label: label shown in the menu (ASCII only)
    - icon: Blender icon name (optional)
    - poll: optional visibility/enabled predicate (context -> bool)
    """

    operator_cls: Type[bpy.types.Operator]
    menu_label: str
    icon: str = "NONE"
    poll: Optional[Callable[[bpy.types.Context], bool]] = None

