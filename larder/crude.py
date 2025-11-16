"""
CRUDE stands for CRUD-Execution.
It is a method to solve the problem of dealing with complex python objects in an
environment that doesn't natively support these.

The method's trick is to allow the complex object's that we "crudified" to be controlled
via a string key that references the complex object, via a "store" which maps
these string keys to the actual physical object.
This store could be a python dictionary (so in RAM) or any persisting storage system
(files, DB) that is given a `typing.Mapping` interface
(see https://i2mint.github.io/dol/ or https://i2mint.github.io/py2store for
tools to do so).

Take, for instance, a GUI that allows a user to compute some descriptive statistics
of the columns of a table.
The inputs are a table, and one of the following statistics function:
``statistics.mean``, ``statistics.median``, or ``statistics.stdev``.

Python functions are not a type natively handled by GUI, so what can we do?
We can stick a layer between our ``compute_stats(stats_func, table)`` function
and our GUI, endowed with a
``{"mean": statistics.mean``, "median": statistics.median, "std": statistics.stdev}``
mapping. We expose the string keys to the GUI, and map them to the functions before
calling ``compute_stats``.

In the case of the ``table``, we'd probably add a means for the GUI user to upload
tables (say from ``.csv`` or ``.xlsx`` files), storing them under a name of their
choice, then pointing to said table via the name, when they want to execute a
``compute_stats(stats_func, table)``.

These are examples of what we call "crudifying" variables or functions.

Here we therefore offer tools to do this sort of thing;
wrap functions so that the complex arguments can be specified through a string key
that points to the actual python object (which is stored in a session's memory or
persisted in some fashion).
"""

from typing import Any, Literal, Optional, Union
from collections.abc import Mapping, Callable, Iterable
from inspect import Parameter
import os
from functools import partial, wraps
import time
from numbers import Number
import asyncio

from i2 import Sig, double_up_as_factory, call_forgivingly
from i2.wrapper import Ingress, wrap
from dol import Files, wrap_kvs
from dol.filesys import mk_tmp_dol_dir, ensure_dir

KT = str
VT = Any
StoreType = Mapping[KT, VT]
StoreName = KT
Mall = Mapping[StoreName, StoreType]


def auto_key_from_arguments(*args, **kwargs) -> KT:
    args_str = ",".join(map(str, args))
    kwargs_str = ",".join(map(lambda kv: f"{kv[0]}={kv[1]}", kwargs.items()))
    return ",".join(filter(None, [args_str, kwargs_str]))


auto_key = auto_key_from_arguments


def auto_key_from_time(*args, __format: Number | str | Callable = 1e6, **kwargs) -> KT:
    utc_seconds = time.time()
    if isinstance(__format, Number):
        return f"{int(utc_seconds * __format):_}"
    elif isinstance(__format, str):
        return time.strftime(__format, time.gmtime(utc_seconds))
    else:
        assert callable(__format)
        return str(__format(utc_seconds))


def _validate_function_keyword_only_params(func, allowed_params: Iterable, obj_name):
    if func is not None:
        if not callable(func):
            raise TypeError(f"{obj_name} should be callable: {func}")
        sig = Sig(func)
        if not all(kind == Parameter.KEYWORD_ONLY for kind in sig.kinds.values()):
            raise TypeError(f"All params of {obj_name} must be keyword-only. {sig}")
        if not set(sig.names).issubset(allowed_params):
            raise TypeError(f"All params of {obj_name} must be in {allowed_params}")


def _store_on_output_sync(*args, _store_on_ouput_args, **kwargs):
    (
        func,
        sig,
        store,
        store_multi_values,
        save_name_param,
        empty_name_callback,
        auto_namer,
        output_trans,
        fixed_save_name,
    ) = _store_on_ouput_args
    arguments = sig.map_arguments(args, kwargs, apply_defaults=True)
    save_name = (
        fixed_save_name
        if fixed_save_name is not None
        else arguments.pop(save_name_param, None)
    )
    # remove save_name from arguments when fixed_save_name supplied so underlying func
    # doesn't receive it as unexpected kwarg
    if fixed_save_name is not None:
        arguments.pop(save_name_param, None)
    if not save_name and empty_name_callback:
        assert callable(empty_name_callback)
        empty_name_callback()
    args, kwargs = sig.mk_args_and_kwargs(arguments)
    output = func(*args, **kwargs)
    if store_multi_values:
        if not isinstance(output, Iterable) or save_name or not auto_namer:
            raise TypeError(
                "When store_multi_values is True, output must be Iterable, save_name must be None and auto_namer must be defined."
            )
        for output_item in output:
            save_name = call_forgivingly(
                auto_namer, arguments=arguments, output=output_item
            )
            store[save_name] = output_item
    else:
        if not save_name and auto_namer:
            save_name = call_forgivingly(auto_namer, arguments=arguments, output=output)
        if save_name:
            store[save_name] = output
    if not output_trans:
        return output
    else:
        return call_forgivingly(
            output_trans, save_name=save_name, output=output, arguments=arguments
        )


async def _store_on_output_async(*args, _store_on_ouput_args, **kwargs):
    (
        func,
        sig,
        store,
        store_multi_values,
        save_name_param,
        empty_name_callback,
        auto_namer,
        output_trans,
        fixed_save_name,
    ) = _store_on_ouput_args
    arguments = sig.map_arguments(args, kwargs, apply_defaults=True)
    save_name = (
        fixed_save_name
        if fixed_save_name is not None
        else arguments.pop(save_name_param, None)
    )
    if fixed_save_name is not None:
        arguments.pop(save_name_param, None)
    if not save_name and empty_name_callback:
        empty_name_callback()
    args, kwargs = sig.mk_args_and_kwargs(arguments)
    output = await func(*args, **kwargs)
    if store_multi_values:
        if not isinstance(output, Iterable) or save_name or not auto_namer:
            raise TypeError(
                "When store_multi_values is True, output must be Iterable, save_name must be None and auto_namer must be defined."
            )
        for output_item in output:
            if asyncio.iscoroutinefunction(auto_namer):
                computed_name = await auto_namer(
                    arguments=arguments, output=output_item
                )
            else:
                computed_name = call_forgivingly(
                    auto_namer, arguments=arguments, output=output_item
                )
            save_name = computed_name
            store[save_name] = output_item
    else:
        if not save_name and auto_namer:
            if asyncio.iscoroutinefunction(auto_namer):
                save_name = await auto_namer(arguments=arguments, output=output)
            else:
                save_name = call_forgivingly(
                    auto_namer, arguments=arguments, output=output
                )
        if save_name:
            store[save_name] = output
    if not output_trans:
        return output
    else:
        if asyncio.iscoroutinefunction(output_trans):
            return await output_trans(
                save_name=save_name, output=output, arguments=arguments
            )
        else:
            return call_forgivingly(
                output_trans, save_name=save_name, output=output, arguments=arguments
            )


@double_up_as_factory
def store_on_output(
    save_name_or_func=None,
    *,
    store=None,
    store_multi_values=False,
    save_name_param="save_name",
    add_store_to_func_attr="output_store",
    empty_name_callback: Callable[[], Any] = None,
    auto_namer: Callable[..., str] = None,
    output_trans: Callable[..., Any] = None,
):
    _validate_function_keyword_only_params(
        auto_namer, ["output", "arguments"], obj_name="auto_namer"
    )
    _validate_function_keyword_only_params(
        output_trans, ["save_name", "output", "arguments"], obj_name="output_trans"
    )
    if output_trans:
        assert callable(output_trans) and set(Sig(output_trans).names).issubset(
            ["save_name", "output", "arguments"]
        )
    fixed_save_name = None

    def _decorate(func):
        sig = Sig(func)
        if save_name_param:
            save_name_param_obj = Parameter(
                name=save_name_param,
                kind=Parameter.KEYWORD_ONLY,
                default="",
                annotation=str,
            )
            sig = sig + [save_name_param_obj]
        elif not auto_namer:
            raise ValueError(
                "There is no way to determine the key under which to store the output. Set a value for the `save_name_param` or `auto_namer` parameters."
            )
        if store is None:
            _store = dict()
        else:
            _store = store
        if asyncio.iscoroutinefunction(func):
            __func = partial(
                _store_on_output_async,
                _store_on_ouput_args=(
                    func,
                    sig,
                    _store,
                    store_multi_values,
                    save_name_param,
                    empty_name_callback,
                    auto_namer,
                    output_trans,
                    fixed_save_name,
                ),
            )
        else:
            __func = partial(
                _store_on_output_sync,
                _store_on_ouput_args=(
                    func,
                    sig,
                    _store,
                    store_multi_values,
                    save_name_param,
                    empty_name_callback,
                    auto_namer,
                    output_trans,
                    fixed_save_name,
                ),
            )
        new_return_annotation = (
            Sig(output_trans).return_annotation if output_trans else Parameter.empty
        )
        if new_return_annotation != Parameter.empty:
            sig = sig.replace(return_annotation=new_return_annotation)
        _func = sig(wraps(func)(__func))
        if isinstance(add_store_to_func_attr, str):
            setattr(_func, add_store_to_func_attr, _store)
        return _func

    # Determine whether decorator was used as @store_on_output or @store_on_output('name', ...)
    if callable(save_name_or_func):
        # used as @store_on_output
        return _decorate(save_name_or_func)
    else:
        fixed_save_name = save_name_or_func

        # used as @store_on_output('name', ...)
        def _decorate_with_name(func):
            return _decorate(func)

        return _decorate_with_name
    if save_name_param:
        save_name_param_obj = Parameter(
            name=save_name_param,
            kind=Parameter.KEYWORD_ONLY,
            default="",
            annotation=str,
        )
        sig = sig + [save_name_param_obj]
    elif not auto_namer:
        raise ValueError(
            "There is no way to determine the key under which to store the output. Set a value for the `save_name_param` or `auto_namer` parameters."
        )
    if store is None:
        store = dict()


@double_up_as_factory
def prepare_for_crude_dispatch(
    func: Callable = None,
    *,
    param_to_mall_map: dict | Iterable | None = None,
    mall: Mall | None = None,
    include_stores_attribute: bool = False,
    output_store: Mapping | str | None = None,
    store_multi_values: bool = False,
    save_name_param: str = "save_name",
    empty_name_callback: Callable[[], Any] = None,
    auto_namer: Callable[..., str] = None,
    output_trans: Callable[..., Any] = None,
    verbose: bool = True,
):
    ingress = None
    store_for_param = {}
    if param_to_mall_map is not None:
        if isinstance(param_to_mall_map, str):
            param_to_mall_map = param_to_mall_map.strip().split()
        param_to_mall_map = keys_to_values_if_non_mapping_iterable(param_to_mall_map)
        sig = Sig(func)
        store_for_param = (
            _mk_store_for_param(sig, param_to_mall_map, mall, verbose=verbose) or dict()
        )

        def kwargs_trans(outer_kw):
            def get_values_from_stores():
                for param, store in store_for_param.items():
                    store_key = outer_kw[param]
                    if isinstance(store_key, str):
                        yield param, store.get(store_key)
                    elif isinstance(store_key, dict):
                        yield param, store.get(store_key)
                    elif isinstance(store_key, Iterable):
                        yield param, [store.get(k) for k in store_key]

            return dict(get_values_from_stores())

        outer_sig = sig.ch_annotations(
            **{param: Literal[tuple(store)] for param, store in store_for_param.items()}
        )
        ingress = Ingress(inner_sig=sig, kwargs_trans=kwargs_trans, outer_sig=outer_sig)
    wrapped_f = wrap(func, ingress=ingress)
    if include_stores_attribute:
        wrapped_f.param_to_mall_map = param_to_mall_map
        wrapped_f.store_for_param = store_for_param
    if output_store is not None:
        output_store_name = "output_store"
        if isinstance(output_store, str):
            output_store_name = output_store
            output_store = mall[output_store_name]
        else:
            if not hasattr(output_store, "__setitem__"):
                raise ValueError(f"Needs to have a __setitem__: {output_store}")
        if output_store_name in store_for_param:
            raise ValueError(
                f"Name conflicts with existing param name: {output_store_name}"
            )
        wrapped_f = store_on_output(
            wrapped_f,
            store=output_store,
            store_multi_values=store_multi_values,
            save_name_param=save_name_param,
            add_store_to_func_attr="output_store" if include_stores_attribute else None,
            empty_name_callback=empty_name_callback,
            auto_namer=auto_namer,
            output_trans=output_trans,
        )
    return wrapped_f


def _mk_store_for_param(sig, param_to_mall_key_dict=None, mall=None, verbose=True):
    mall = mall or dict()
    unmentioned_mall_keys = set(mall) & set(sig.names) - set(param_to_mall_key_dict)
    if unmentioned_mall_keys and verbose:
        from warnings import warn

        warn(
            f"Some of your mall keys were also func arg names, but you didn't mention them in param_to_mall_map, namely, these: {unmentioned_mall_keys}"
        )
    if param_to_mall_key_dict:
        if isinstance(param_to_mall_key_dict, str):
            param_to_mall_key_dict = param_to_mall_key_dict.split()
        if not set(param_to_mall_key_dict).issubset(sig.names):
            offenders = set(param_to_mall_key_dict) - set(sig.names)
            raise ValueError(
                f"The param_to_mall_map should only contain keys that are parameters of your function. Offenders: {offenders}"
            )
        if not set(param_to_mall_key_dict.values()).issubset(mall.keys()):
            offenders = set(param_to_mall_key_dict.values()) - set(mall.keys())
            keys = "keys" if len(offenders) > 1 else "key"
            offenders = ", ".join(map(lambda x: f"'{x}'", offenders))
            raise ValueError(
                f"The {offenders} {keys} of your param_to_mall_map values were not in the mall."
            )
        store_for_param = {
            argname: mall[mall_key]
            for argname, mall_key in param_to_mall_key_dict.items()
        }
        return store_for_param


def keys_to_values_if_non_mapping_iterable(d: Iterable | None) -> dict:
    if d is None:
        return dict()
    elif not isinstance(d, Mapping) and isinstance(d, Iterable):
        d = {k: k for k in d}
    return d


def simple_mall_dispatch_core_func(
    key: KT, action: str, store_name: StoreName, mall: Mall
):
    if not store_name:
        return list(mall)
    else:
        store = mall[store_name]
        if not action:
            return store
    key = key or ""
    if action == "list":
        key = key.strip()
        return list(filter(lambda k: key in k, store))
    elif action == "get":
        return store[key]


from collections.abc import Iterable
from i2 import Sig, Pipe
from i2.signatures import sig_to_dataclass

_Crudifier = sig_to_dataclass(
    Sig(prepare_for_crude_dispatch).params[1:], cls_name="_Crudifier", module=__name__
)


class Crudifier(_Crudifier):
    def __call__(self, func):
        return prepare_for_crude_dispatch(func, **vars(self))


from i2 import name_of_obj
from dol import chain_get


def _remove_non_valued_items(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def _keys_to_search(func):
    func_name = name_of_obj(func)
    for arg_name in Sig(func).names:
        yield (
            arg_name,
            (
                (func, arg_name),
                (func_name, arg_name),
                f"{func_name}.{arg_name}",
                arg_name,
            ),
        )


def crudify_based_on_names(
    func, *, param_to_mall_map=(), output_store=(), crudifier=Crudifier
):
    param_to_mall_map = dict(param_to_mall_map)
    output_store = dict(output_store)
    func_name = name_of_obj(func)
    param_to_mall_map = (
        _remove_non_valued_items(
            {
                arg_name: chain_get(param_to_mall_map, keys)
                for arg_name, keys in _keys_to_search(func)
            }
        )
        or None
    )
    output_store = chain_get(output_store, (func, func_name), default=None)
    if param_to_mall_map or output_store:
        return crudifier(
            func, param_to_mall_map=param_to_mall_map, output_store=output_store
        )
    else:
        return func


try:
    import dill as pickler
except ImportError:
    import pickle as pickler
    from warnings import warn

    warn("Could not import dill. The DillFiles will use pickle instead.")


@wrap_kvs(data_of_obj=pickler.dumps, obj_of_data=pickler.loads)
class DillFiles(Files):
    pass


def mk_mall_of_dill_stores(store_names=Iterable[StoreName], rootdir=None):
    rootdir = rootdir or mk_tmp_dol_dir("crude")
    if isinstance(store_names, str):
        store_names = store_names.split()

    def name_and_rootdir():
        for name in store_names:
            root = os.path.join(rootdir, name)
            ensure_dir(root)
            yield name, root

    return {name: DillFiles(root) for name, root in name_and_rootdir()}
