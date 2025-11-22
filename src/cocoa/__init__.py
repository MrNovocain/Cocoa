"""
Top-level package for the Cocoa price forecasting / structural-break project.
Provides convenient access to global settings and paths.
"""

from .config import settings
from .paths import PATHS

__all__ = ["settings", "PATHS"]
