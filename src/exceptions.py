"""
Custom Exception Classes for the Application
"""


class WakewordException(Exception):
    """Base class for exceptions in this application."""


class DataLoadError(WakewordException):
    """Exception raised for errors in loading data."""


class AudioProcessingError(WakewordException):
    """Exception raised for errors in audio processing."""


class ModelLoadError(WakewordException):
    """Exception raised for errors in loading a model."""


class ConfigurationError(WakewordException):
    """Exception raised for errors in configuration."""
