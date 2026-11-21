# larder.dog

DOG (Data Operation Graph)

### Classes

| [`ADOG`](#larder.dog.ADOG)(operation_signatures, data_stores, ...)   | Asynchronous DOG: like [`DOG`](#larder.dog.DOG) but runs operations via `async_compute`.   |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [`DOG`](#larder.dog.DOG)(operation_signatures, data_stores, ...)    | Synchronous DAG-of-operations executor over typed data stores.                                                         |

### *class* larder.dog.ADOG(operation_signatures, data_stores, operation_implementations, , base_path=None, ttl_seconds=3600, serialization=SerializationFormat.JSON, middleware=None, sourced_argnames=None)

Bases: `_DOG`

Asynchronous DOG: like [`DOG`](#larder.dog.DOG) but runs operations via `async_compute`.

Adds caching/persistence knobs (`ttl_seconds`, `serialization`,
`middleware`) and a `base_path` for the async result stores; otherwise
the operation/store/signature model is the same as [`DOG`](#larder.dog.DOG).

#### call(func_impl, \*args, \*\*kwargs)

Run `func_impl` via its cached async wrapper, persisting the output.

Returns `(store_name, output_key)`; raises `ValueError` if no store
matches the operation’s return type.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* larder.dog.DOG(operation_signatures, data_stores, operation_implementations, sourced_argnames=None)

Bases: `_DOG`

Synchronous DAG-of-operations executor over typed data stores.

Given `operation_signatures` (`Callable[[In], Out]` per operation),
`data_stores` (each tagged with a `type`), and
`operation_implementations`, [`call()`](#larder.dog.DOG.call) runs an implementation, sources
any string-key arguments from the stores, and persists the output to the
store whose `type` matches the operation’s return type. A runnable
example is in the README (“DOG example”).

#### call(func_impl, \*args, \*\*kwargs)

Run `func_impl`, persist its output, and return `(store_name, key)`.

Raises `ValueError` if no store matches the operation’s return type.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
