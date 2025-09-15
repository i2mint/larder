"""
Value Dispatch using Dependency Injector Package
"""

from functools import wraps
import inspect


def input_wiring(func, global_param_to_store, mall=None):
    if mall is None:
        raise ValueError("A mall must be provided to access storage systems.")
    sig = inspect.signature(func)
    sig_names = sig.parameters.keys()
    param_to_store = {
        param: global_param_to_store[param]
        for param in sig_names
        if param in global_param_to_store
    }

    def resolve_bound_args(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param in param_to_store:
            key = bound_args.arguments[param]
            store_name = param_to_store[param]
            store = getattr(mall, store_name)()
            if key not in store:
                raise KeyError(f"Key '{key}' not found in store '{store_name}'")
            bound_args.arguments[param] = store[key]
        return bound_args

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            bound_args = resolve_bound_args(*args, **kwargs)
            result = await func(*bound_args.args, **bound_args.kwargs)
            return result

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = resolve_bound_args(*args, **kwargs)
            result = func(*bound_args.args, **bound_args.kwargs)
            return result

    return wrapper
