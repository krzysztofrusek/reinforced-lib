from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from chex import Array, Scalar

from reinforced_lib.utils.exceptions import UnsupportedLogTypeError


class SourceType(Enum):
    OBSERVATION = 0
    STATE = 1
    METRIC = 2


Source = Union[Tuple[str, SourceType], str, None]


class BaseLogger(ABC):
    """
    Container for functions of a logger. Provides a simple interface for defining custom loggers.
    """

    def __init__(self, **kwargs):
        pass

    def init(self, sources: List[Source]) -> None:
        """
        Initializes the logger given the list of all sources.

        Parameters
        ----------
        sources : list[Source]
            List containing the sources to log.
        """

        pass

    def finish(self) -> None:
        """
        Finalizes the loggers work, for example, saves data or shows plots.
        """

        pass

    def log_scalar(self, source: Source, value: Scalar, custom: bool) -> None:
        """
        Method of the logger interface used for logging scalar values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        custom : bool
            Whether the source is a custom source.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_array(self, source: Source, value: Array, custom: bool) -> None:
        """
        Method of the logger interface used for logging arrays.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        custom : bool
            Whether the source is a custom source.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_dict(self, source: Source, value: Dict, custom: bool) -> None:
        """
        Method of the logger interface used for logging dictionaries.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        custom : bool
            Whether the source is a custom source.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_other(self, source: Source, value: Any, custom: bool) -> None:
        """
        Method of the logger interface used for logging other values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        custom : bool
            Whether the source is a custom source.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    @staticmethod
    def source_to_name(source: Source) -> str:
        """
        Converts the source to a string. If source is a string itself, it returns that string.
        Otherwise, it returns a string in the format "name-sourcetype" (e.g., "action-metric").

        Parameters
        ----------
        source : Source
            Source of the logged value.

        Returns
        -------
        str
            Name of the source.
        """

        if isinstance(source, tuple):
            return f'{source[0]}-{source[1].name.lower()}'
        else:
            return source
