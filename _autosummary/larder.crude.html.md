# larder.crude

CRUDE stands for CRUD-Execution.
It is a method to solve the problem of dealing with complex python objects in an
environment that doesn’t natively support these.

The method’s trick is to allow the complex object’s that we “crudified” to be controlled
via a string key that references the complex object, via a “store” which maps
these string keys to the actual physical object.
This store could be a python dictionary (so in RAM) or any persisting storage system
(files, DB) that is given a `typing.Mapping` interface
(see [https://i2mint.github.io/dol/](https://i2mint.github.io/dol/) or [https://i2mint.github.io/py2store](https://i2mint.github.io/py2store) for
tools to do so).

Take, for instance, a GUI that allows a user to compute some descriptive statistics
of the columns of a table.
The inputs are a table, and one of the following statistics function:
`statistics.mean`, `statistics.median`, or `statistics.stdev`.

Python functions are not a type natively handled by GUI, so what can we do?
We can stick a layer between our `compute_stats(stats_func, table)` function
and our GUI, endowed with a
`{"mean": statistics.mean`, “median”: statistics.median, “std”: statistics.stdev}\`\`
mapping. We expose the string keys to the GUI, and map them to the functions before
calling `compute_stats`.

In the case of the `table`, we’d probably add a means for the GUI user to upload
tables (say from `.csv` or `.xlsx` files), storing them under a name of their
choice, then pointing to said table via the name, when they want to execute a
`compute_stats(stats_func, table)`.

These are examples of what we call “crudifying” variables or functions.

Here we therefore offer tools to do this sort of thing;
wrap functions so that the complex arguments can be specified through a string key
that points to the actual python object (which is stored in a session’s memory or
persisted in some fashion).

### Functions

| [`auto_key`](#larder.crude.auto_key)(\*args, \*\*kwargs)                     | Build a string key from a call's positional and keyword arguments.              |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`auto_key_from_arguments`](#larder.crude.auto_key_from_arguments)(\*args, \*\*kwargs)      | Build a string key from a call's positional and keyword arguments.              |
| [`auto_key_from_time`](#larder.crude.auto_key_from_time)(\*args[, \_\_format])         | Make a string key from the current UTC time.                                    |
| [`crudify_based_on_names`](#larder.crude.crudify_based_on_names)(func, \*[, ...])          | Crudify `func` by matching its parameter names against name-keyed maps.         |
| [`keys_to_values_if_non_mapping_iterable`](#larder.crude.keys_to_values_if_non_mapping_iterable)(d)        | Normalize `d` to a mapping: a non-mapping iterable becomes `{x: x}`.            |
| [`mk_mall_of_dill_stores`](#larder.crude.mk_mall_of_dill_stores)([store_names, rootdir])   | Make a mall (dict of stores) of dill-backed file stores, one per name.          |
| `prepare_for_crude_dispatch`([func, ...])                                                         | Adapt a function for "crude" dispatch: source string-key arguments from a mall. |
| [`simple_mall_dispatch_core_func`](#larder.crude.simple_mall_dispatch_core_func)(key, action, ...) | Dispatch a CRUD-ish read against a mall: list stores, list/get keys.            |
| `store_on_output`([save_name_or_func, store, ...])                                                | Decorate a function so its return value is (optionally) persisted to a store.   |

### Classes

| [`Crudifier`](#larder.crude.Crudifier)([param_to_mall_map, mall, ...])   | A reusable, configured `prepare_for_crude_dispatch` as a dataclass.   |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`DillFiles`](#larder.crude.DillFiles)(\*args[, delete_func, ...])       | A `dol.Files` store that serializes values with `dill` (pickle++).    |

### *class* larder.crude.Crudifier(param_to_mall_map=None, mall=None, include_stores_attribute=False, output_store=None, store_multi_values=False, save_name_param='save_name', empty_name_callback=None, auto_namer=None, output_trans=None, verbose=True)

Bases: `_Crudifier`

A reusable, configured `prepare_for_crude_dispatch` as a dataclass.

Holds the crudification options (`param_to_mall_map`, `output_store`,
…) as fields, so one configured instance can crudify many functions:
`crudify = Crudifier(mall=mall, param_to_mall_map=...); crudify(func)`.

### *class* larder.crude.DillFiles(\*args, delete_func=None, write_open_kwargs=None, write_func=None, \*\*kwargs)

Bases: `Store`

A `dol.Files` store that serializes values with `dill` (pickle++).

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### larder.crude.auto_key(\*args, \*\*kwargs)

Build a string key from a call’s positional and keyword arguments.

```pycon
>>> auto_key_from_arguments(1, 2, a=3)
'1,2,a=3'
>>> auto_key_from_arguments(a=1, b=2)
'a=1,b=2'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### larder.crude.auto_key_from_arguments(\*args, \*\*kwargs)

Build a string key from a call’s positional and keyword arguments.

```pycon
>>> auto_key_from_arguments(1, 2, a=3)
'1,2,a=3'
>>> auto_key_from_arguments(a=1, b=2)
'a=1,b=2'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### larder.crude.auto_key_from_time(\*args, \_\_format=1000000.0, \*\*kwargs)

Make a string key from the current UTC time.

`__format` controls the rendering: a number multiplies the UTC seconds
(default `1e6` → microseconds as a grouped integer string), a `strftime`
format string formats the time, and a callable is passed the UTC seconds.
Ignores `*args`/`**kwargs` so it can be dropped in as an auto-namer.

```pycon
>>> auto_key_from_time(__format=lambda utc_seconds: 'fixed-key')
'fixed-key'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### larder.crude.crudify_based_on_names(func, \*, param_to_mall_map=(), output_store=(), crudifier=<class 'larder.crude.Crudifier'>)

Crudify `func` by matching its parameter names against name-keyed maps.

Looks each parameter up in `param_to_mall_map` (and the output in
`output_store`) by the function object, its name, `name.arg`, or the
bare arg name — so a single name-indexed config can crudify many functions.
Returns `func` unchanged when nothing matches.

### larder.crude.keys_to_values_if_non_mapping_iterable(d)

Normalize `d` to a mapping: a non-mapping iterable becomes `{x: x}`.

A mapping (or `None`) is returned as a dict unchanged; an iterable of
items is turned into an identity mapping.

```pycon
>>> keys_to_values_if_non_mapping_iterable(['a', 'b'])
{'a': 'a', 'b': 'b'}
>>> keys_to_values_if_non_mapping_iterable(None)
{}
>>> keys_to_values_if_non_mapping_iterable({'a': 1})
{'a': 1}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### larder.crude.mk_mall_of_dill_stores(store_names=collections.abc.Iterable[str], rootdir=None)

Make a mall (dict of stores) of dill-backed file stores, one per name.

`store_names` may be an iterable of names or a space-separated string;
each store persists under `rootdir/<name>` (a temp dir by default).

### larder.crude.simple_mall_dispatch_core_func(key, action, store_name, mall)

Dispatch a CRUD-ish read against a mall: list stores, list/get keys.

With no `store_name`, returns the mall’s store names. With a store but no
`action`, returns the store. `action='list'` returns the keys
containing `key` (a substring filter); `action='get'` returns
`store[key]`.

```pycon
>>> mall = {'s1': {'apple': 1, 'apricot': 2, 'banana': 3}}
>>> sorted(simple_mall_dispatch_core_func(None, None, None, mall))
['s1']
>>> simple_mall_dispatch_core_func('ap', 'list', 's1', mall)
['apple', 'apricot']
>>> simple_mall_dispatch_core_func('banana', 'get', 's1', mall)
3
```
