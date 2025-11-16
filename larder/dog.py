"""
DOG (Data Operation Graph)
"""

from typing import Any, Dict, List, Tuple, get_args
from collections.abc import Callable
import os
import tempfile

from au.base import async_compute, FileSystemStore, SerializationFormat


class _DOG:
    def __init__(
        self,
        operation_signatures: dict[str, Any],
        data_stores: dict[str, Any],
        operation_implementations: dict[str, Any],
        sourced_argnames: Any = None,
    ):
        self.operation_signatures = operation_signatures
        self.operation_implementations = operation_implementations
        self._return_type_to_store_name_map = {}
        for store_name, store_config in data_stores.items():
            data_type_class = store_config["type"]
            self._return_type_to_store_name_map[data_type_class] = store_name
        self.data_stores = {
            name: config["store"] for name, config in data_stores.items()
        }
        if sourced_argnames is True:
            sourced_argnames = {k: k for k in self.data_stores.keys()}
        elif not sourced_argnames:
            sourced_argnames = {}
        else:
            sourced_argnames = dict(sourced_argnames)
        self.sourced_argnames = self._validate_sourced_argnames(sourced_argnames)
        self._output_counter = 0

    def _validate_sourced_argnames(self, sourced_argnames):
        for arg, store in sourced_argnames.items():
            if store not in self.data_stores:
                raise ValueError(
                    f"sourced_argnames: store '{store}' for argument '{arg}' is not a valid data store name"
                )
        return sourced_argnames

    def _source_args(self, func_impl, args, kwargs):
        if not self.sourced_argnames:
            return args, kwargs
        import inspect

        sig = inspect.signature(func_impl)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for argname, storename in self.sourced_argnames.items():
            if argname in bound.arguments:
                val = bound.arguments[argname]
                store = self.data_stores[storename]
                if isinstance(val, str) and val in store:
                    bound.arguments[argname] = store[val]
        return bound.args, bound.kwargs

    def _get_output_store_name_and_type(self, func_impl: Callable):
        output_store_name = None
        return_type_class = None
        for op_type_name, impl_dict in self.operation_implementations.items():
            if func_impl in impl_dict.values():
                signature_callable = self.operation_signatures.get(op_type_name)
                if signature_callable:
                    signature_args = get_args(signature_callable)
                    if signature_args:
                        return_type_class = signature_args[-1]
                    break
        if return_type_class:
            output_store_name = self._return_type_to_store_name_map.get(
                return_type_class
            )
        return output_store_name, return_type_class

    def _get_next_output_key(self, output_store_name):
        self._output_counter += 1
        return f"output_{output_store_name}_{self._output_counter}.json"


class DOG(_DOG):
    def call(self, func_impl: Callable, *args, **kwargs) -> tuple[str, str]:
        args, kwargs = self._source_args(func_impl, args, kwargs)
        output_store_name, _ = self._get_output_store_name_and_type(func_impl)
        if not output_store_name or output_store_name not in self.data_stores:
            raise ValueError(
                f"Could not determine a valid output data store for function implementation: {func_impl}."
            )
        output_data = func_impl(*args, **kwargs)
        output_key = self._get_next_output_key(output_store_name)
        self.data_stores[output_store_name][output_key] = output_data
        return output_store_name, output_key


class ADOG(_DOG):
    def __init__(
        self,
        operation_signatures: dict[str, Any],
        data_stores: dict[str, Any],
        operation_implementations: dict[str, Any],
        *,
        base_path: str = None,
        ttl_seconds: int = 3600,
        serialization: SerializationFormat = SerializationFormat.JSON,
        middleware: list[Any] = None,
        sourced_argnames: dict[str, str] = None,
    ):
        super().__init__(
            operation_signatures,
            data_stores,
            operation_implementations,
            sourced_argnames=sourced_argnames,
        )
        if base_path is None:
            base_path = tempfile.mkdtemp(prefix="adog_store_")
        self._adog_base_path = base_path
        self._adog_ttl_seconds = ttl_seconds
        self._adog_serialization = serialization
        self._adog_middleware = middleware or []
        self._async_wrappers = {}
        output_types = set()
        for op_signature in self.operation_signatures.values():
            signature_args = get_args(op_signature)
            if signature_args:
                return_type = signature_args[-1]
                output_types.add(return_type)
        output_store_names = set()
        for output_type in output_types:
            store_name = self._return_type_to_store_name_map.get(output_type)
            if store_name:
                output_store_names.add(store_name)
        for store_name in output_store_names:
            store = self.data_stores[store_name]
            if not isinstance(store, FileSystemStore):
                store_path = os.path.join(self._adog_base_path, store_name)
                os.makedirs(store_path, exist_ok=True)
                self.data_stores[store_name] = FileSystemStore(
                    store_path,
                    ttl_seconds=self._adog_ttl_seconds,
                    serialization=self._adog_serialization,
                )

    def call(self, func_impl: Callable, *args, **kwargs) -> tuple[str, str]:
        args, kwargs = self._source_args(func_impl, args, kwargs)
        output_store_name, _ = self._get_output_store_name_and_type(func_impl)
        if not output_store_name or output_store_name not in self.data_stores:
            raise ValueError(
                f"Could not determine a valid output data store for function implementation: {func_impl}."
            )
        output_store = self.data_stores[output_store_name]
        if func_impl not in self._async_wrappers:
            async_func = async_compute(
                store=output_store,
                ttl_seconds=self._adog_ttl_seconds,
                serialization=self._adog_serialization,
                middleware=self._adog_middleware,
            )(func_impl)
            self._async_wrappers[func_impl] = async_func
        else:
            async_func = self._async_wrappers[func_impl]
        handle = async_func(*args, **kwargs)
        return output_store_name, handle.key
