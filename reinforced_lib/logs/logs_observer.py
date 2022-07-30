from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import jax.numpy as jnp

from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.logs.base_logger import BaseLogger


class SourceType(int, Enum):
    OBSERVATIONS = 0
    AGENT_STATE = 1
    METRICS = 2


Source = Union[Tuple[str, SourceType], str]


class LogsObserver:
    def __init__(self) -> None:
        self._loggers_instances: Dict[type, BaseLogger] = {}
        self._loggers_sources: Dict[BaseLogger, List[Source]] = defaultdict(list)

        self._observations_loggers: Dict[BaseLogger, List[str]] = defaultdict(list)
        self._agent_state_loggers: Dict[BaseLogger, List[str]] = defaultdict(list)
        self._metrics_loggers: Dict[BaseLogger, List[str]] = defaultdict(list)

    def add_logger(self, source: Source, logger_type: type, logger_params: Dict[str, Any]) -> None:
        logger = self._get_logger(logger_type, logger_params)

        if isinstance(source, tuple):
            if source[1] == SourceType.OBSERVATIONS:
                self._observations_loggers[logger].append(source[0])
            elif source[1] == SourceType.AGENT_STATE:
                self._agent_state_loggers[logger].append(source[0])
            elif source[1] == SourceType.METRICS:
                self._metrics_loggers[logger].append(source[0])
        elif isinstance(source, str):
            self._observations_loggers[logger].append(source)
            self._agent_state_loggers[logger].append(source)
            self._metrics_loggers[logger].append(source)

        self._loggers_sources[logger].append(source)

    def _get_logger(self, ltype: type, logger_params: Dict[str, Any]) -> BaseLogger:
        if ltype not in self._loggers_instances:
            self._loggers_instances[ltype] = ltype(**logger_params)

        return self._loggers_instances[ltype]

    def update_observations(self, observations: Any) -> None:
        if isinstance(observations, dict):
            self._update(self._observations_loggers, lambda name: observations.get(name, None))

    def update_agent_state(self, agent_state: BaseAgent) -> None:
        self._update(self._agent_state_loggers, lambda name: getattr(agent_state, name, None))

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        self._update(self._metrics_loggers, lambda name: metrics.get(name, None))

    @staticmethod
    def _update(loggers: Dict[BaseLogger, List[str]], get_value: Callable) -> None:
        for logger, names in loggers.items():
            for name in names:
                if (value := get_value(name)) is not None:
                    if jnp.isscalar(value):
                        logger.log_scalar(name, value)
                    elif isinstance(value, dict):
                        logger.log_dict(name, value)
                    elif hasattr(value, '__len__'):
                        logger.log_array(name, value)
                    else:
                        logger.log_other(name, value)

    def init_loggers(self):
        for logger, sources in self._loggers_sources.items():
            logger.init(sources)

    def finish_loggers(self):
        for logger in self._loggers_sources.keys():
            logger.finish()
