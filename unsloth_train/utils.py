# coding: utf-8

from inspect import signature
from typing import Any, Callable, Dict


def _get_function_used_params(
    callable: Callable,
    **kwds: Dict,
) -> Dict[str, Any]:
    """Get `callable` need parameters from kwds.

    Args:
        callable (Callable): function

    Returns:
        Dict[str, Any]: parameters
    """
    parameters = dict()
    callable_parameters = signature(callable).parameters
    for parameter, value in kwds.items():
        if parameter in callable_parameters:
            if value:
                parameters.update({parameter: value})
            else:
                parameters.update({parameter: None})
    return parameters
