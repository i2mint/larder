"""
Input sourcing functionality for the source_variables decorator pattern.

This module provides flexible variable resolution for API endpoints through
the source_variables decorator, which allows functions to resolve their
inputs from various stores and data sources.
"""

import asyncio
from functools import partial, wraps
from typing import Dict, Any, Callable


def resolve_data(data, store_key, mall):
    if isinstance(data, str) and store_key in mall and data in mall[store_key]:
        return mall[store_key][data]
    return data


async def _get_function_from_store(key, store_key, mall):
    if store_key not in mall:
        raise KeyError(f"Store '{store_key}' not found in mall")
    store = mall[store_key]
    if isinstance(key, dict):
        if len(key) != 1:
            raise KeyError(f"Dict key must contain exactly one item, got {len(key)}")
        func_key = next(iter(key))
        func_kwargs = key[func_key]
        if func_key not in store:
            raise KeyError(f"Key '{func_key}' not found in store '{store_key}'")
        base_func = store[func_key]
        return partial(base_func, **func_kwargs)
    if key not in store:
        raise KeyError(f"Key '{key}' not found in store '{store_key}'")
    return store[key]


def source_variables(__var_store_suffix="s", **config):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            mall_provider = config.get("mall", lambda: mock_mall)
            egress_func = config.get("egress")
            mall = mall_provider() if callable(mall_provider) else mall_provider
            for var_name, var_config in config.items():
                if var_name in ("mall", "egress"):
                    continue
                if var_name not in kwargs:
                    continue
                if callable(var_config):
                    resolver = var_config
                    store_key = var_name + __var_store_suffix
                    mode = "hybrid"
                    condition = lambda x: True
                    ingress = lambda obj, v: obj
                else:
                    resolver = var_config.get("resolver", resolve_data)
                    store_key = var_config.get(
                        "store_key", var_name + __var_store_suffix
                    )
                    mode = var_config.get("mode", "hybrid")
                    condition = var_config.get("condition", lambda x: True)
                    ingress = var_config.get("ingress", lambda obj, v: obj)
                value = kwargs[var_name]
                if condition(value):
                    if asyncio.iscoroutinefunction(resolver):
                        resolved_value = await resolver(value, store_key, mall)
                    else:
                        resolved_value = resolver(value, store_key, mall)
                    if (
                        mode == "store_only"
                        and isinstance(value, str)
                        and resolved_value == value
                    ):
                        raise KeyError(
                            f"Key '{value}' not found in store '{store_key}'"
                        )
                    if isinstance(value, dict) and len(value) == 1:
                        obj_key = next(iter(value))
                        obj_value = value[obj_key]
                        if (
                            isinstance(resolved_value, dict)
                            and obj_key in resolved_value
                        ):
                            base_obj = resolved_value[obj_key]
                            kwargs[var_name] = ingress(base_obj, obj_value)
                        else:
                            kwargs[var_name] = ingress(resolved_value, obj_value)
                    else:
                        kwargs[var_name] = resolved_value
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            if egress_func:
                return egress_func(result)
            return result

        return wrapper

    return decorator


mock_mall = {
    "embedders": {
        "default": lambda text: (
            [ord(c) for c in text] if isinstance(text, str) else text
        ),
        "advanced": lambda text, multiplier=1: (
            [ord(c) * multiplier for c in text]
            if isinstance(text, str)
            else [v * multiplier for v in text]
        ),
    },
    "segments": {"greeting": "hello", "farewell": "goodbye"},
    "clusterers": {
        "default": lambda embeddings: [
            sum(e) % 3
            for e in (embeddings if isinstance(embeddings[0], list) else [embeddings])
        ],
        "binary": lambda embeddings: [
            1 if sum(e) > 500 else 0
            for e in (embeddings if isinstance(embeddings[0], list) else [embeddings])
        ],
    },
}
