"""Shared error types for plmol."""

from __future__ import annotations


class PlmolError(Exception):
    """Base error type for plmol."""


class InputError(PlmolError, ValueError):
    """Raised when user input is invalid or unsupported."""


class DependencyError(PlmolError, ImportError):
    """Raised when an optional or required dependency is missing."""


class FeatureError(PlmolError, RuntimeError):
    """Raised when feature extraction fails at runtime."""
