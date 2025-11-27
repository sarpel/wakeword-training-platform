"""
Custom Exception Classes for the Application
"""


class WakewordException(Exception):
    """Base class for exceptions in this application."""

    pass


class DataLoadError(WakewordException):
    """Exception raised for errors in loading data."""

    pass


class AudioProcessingError(WakewordException):
    """Exception raised for errors in audio processing."""

    pass


class ModelLoadError(WakewordException):
    """Exception raised for errors in loading a model."""

    pass


class ConfigurationError(WakewordException):
    """Exception raised for errors in configuration."""

    pass
