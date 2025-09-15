# larder — connect functions to stores

Connect functions to stores - fetch inputs & persist outputs.

This package provides small utilities to "CRUD-ify" function inputs and outputs:
let callers refer to any complex Python objects (functions, tables, etc.)
by string keys stored in simple mappings (a "mall"), and to persist outputs
into stores. The code is intentionally tiny and dependency-light; it contains
three practical pieces you will use together or separately:

- `crude.py` — helpers to wrap functions so they read inputs from stores and
	persist outputs (decorators, factory wrappers, simple file-backed helpers).
    (Note: "crude" stands for "CRUD Execution")
- `dog.py` — a small Data Operation Graph (DOG) plus an async variant (ADOG)
	which route operation outputs into the appropriate data store.

Why this exists
---------------
Many UIs and service layers cannot pass rich Python objects (functions,
in-memory tables, model objects) directly. Instead they pass lightweight keys
that reference objects stored in a session or filesystem. `larder` gives you
the plumbing to map those keys back into real Python values before calling
your functions, and to persist outputs back into stores under predictable
keys.

This makes it easy to:

- Drive computations from a GUI or web API using human-friendly string keys.
- Persist outputs (results, embeddings, cluster indices) alongside data.
- Swap stores (in-memory dicts, file-backed stores) without changing
	the wrapped business logic.

Key concepts and patterns
-------------------------
- Decorator-as-factory: `store_on_output` and `prepare_for_crude_dispatch` can
	be used either as plain decorators or as factories that accept configuration
	and return a decorator. This lets you write `@store_on_output` or
	`@store_on_output("name", store=...)`.
- Signature reflection: wrappers use `i2.Sig` to keep function signatures
	transparent (so wrapped functions still show sensible parameter lists).
- Mall pattern: a `mall` is a mapping {store_name: store}, where `store` is a
	mapping-like object. It can be a plain dict or a file-backed store (see
	`DillFiles` in `crude.py`).
- Sync/async compatibility: wrappers work for both synchronous and
	asynchronous functions where applicable (the code checks for coroutine
	functions and dispatches accordingly).

Short working examples (taken from the tests)
--------------------------------------------
These examples are simplified extracts from the package tests so you can
quickly try the patterns that are covered by tests in `larder/tests`.

1) Persisting a function's output (sync)

```python
from larder import store_on_output

output_store = {}

@store_on_output('result', store=output_store)
def add(a, b):
		return a + b

res = add(2, 3)
assert res == 5
assert output_store['result'] == 5
```

Why useful: attach simple persistence to pure functions without changing the
function body. Use cases: caching, background result pickling, lightweight
audit logs.

(2) DOG example (minimal)

```python
from larder import DOG

data_stores = {
		'segments': {'type': list, 'store': {'s1': ['a','b']}},
		'embeddings': {'type': list, 'store': {}},
}

ops = {'embedder': {'simple': lambda segs: [[ord(c) for c in s] for s in segs]}}
signatures = {'embedder': callable}

dog = DOG(operation_signatures=signatures, data_stores=data_stores, operation_implementations=ops)
out_store, out_key = dog.call(ops['embedder']['simple'], ['hello'])
assert out_store == 'embeddings'
```

API reference (main interfaces)
--------------------------------
Below are the most important functions and classes you will use. The list is
concise but mentions the main options; consult the code for advanced knobs.

```py
crude.store_on_output(save_name_or_func=None, *, store=None, store_multi_values=False,
											save_name_param='save_name', add_store_to_func_attr='output_store',
											empty_name_callback=None, auto_namer=None, output_trans=None)
```

- Use: decorate functions so their return value is optionally persisted.
- Important args:
	- save_name_or_func: if used as `@store_on_output('name', ...)` this is the
		fixed save name, otherwise used as plain decorator.
	- store: mapping-like object where outputs are saved (default: in-memory dict).
	- store_multi_values: when True, expects the function to return an iterable
		and stores each item individually (requires `auto_namer` and
		`save_name_param=None`).
	- save_name_param: name of the (keyword-only) parameter used to pass
		a save name at call-time (default 'save_name'). If None, call-time naming
		is disabled.
	- auto_namer(arguments=..., output=...): a callable used to compute names.
	- output_trans(save_name=..., output=..., arguments=...): optional callable
		to post-process results (return value of wrapped function replaced by
		output_trans return).

crude.prepare_for_crude_dispatch(func=None, *, param_to_mall_map=None, mall=None,
																 include_stores_attribute=False, output_store=None,
																 store_multi_values=False, save_name_param='save_name',
																 empty_name_callback=None, auto_namer=None, output_trans=None,
																 verbose=True)
- Use: wrap a function so specified parameters are fetched from mall stores
	before calling, and optionally attach `store_on_output` behavior to persist
	outputs.
- Important args:
	- param_to_mall_map: mapping of function parameter names -> mall store keys.
	- mall: the mall mapping (store_name -> store mapping object).
	- output_store: either a store object or a store-name in the mall to persist
		outputs.

crude.DillFiles(root)
- File-backed store that pickles objects. Useful for simple persistence of
	Python objects without setting up a database.

larder.wip.input_sourcing.source_variables(**config)
- Use: decorate functions to resolve specific parameters from stores.
- Important args in `config` for each parameter name:
	- resolver: callable(value, store_key, mall) | async callable
	- store_key: name of the store in the mall
	- mode: 'hybrid' (default) or 'store_only' (raise error if key missing)
	- condition: callable(value) -> bool to decide whether to resolve
	- ingress: optional transformer when resolving parameterized values
- The decorator also accepts `mall` (callable or mapping) and `egress`
	(post-processing of the function result).

larder.wip.input_sourcing.resolve_data(data, store_key, mall)
- Simple helper that returns mall[store_key][data] when `data` is a key.

larder.wip.input_sourcing._get_function_from_store(key, store_key, mall)
- Async helper that returns a function stored under `store_key`; supports
	parameterized function descriptors like `{'name': {'multiplier': 2}}`.

larder.wip.input_wire.input_wiring(func, global_param_to_store, mall=None)
- Use: wrap a function so named parameters are fetched from mall stores using
	a mapping `global_param_to_store` (param -> store_name). The mall object is
	expected to expose callable attributes returning mapping-like stores.

larder.dog.DOG(operation_signatures, data_stores, operation_implementations, sourced_argnames=None)
- Use: small orchestration helper that determines where operation outputs
	should be stored based on declared return types and stores mapping.
- ADOG: async variant that wraps operations with an async compute wrapper and
	file-backed stores when needed.

How to run the tests that back these examples
--------------------------------------------
Run the package tests (they contain the full examples):

```bash
cd /path/to/proj/i/larder/larder
pytest -q
```

Notes and next steps
--------------------
- The package is intentionally small and meant as a focused toolbox rather
	than a full-blown orchestration framework. You can plug any mapping-like
	store into the mall (in-memory dicts, `DillFiles`, or custom file stores).
- If you want, I can add a tiny example script under `examples/` that shows a
	small end-to-end flow (upload -> compute -> persist) using `DillFiles`.

If you'd like the README changed in tone, length, or to include additional
examples (FastAPI, background workers, or real file-backed stores), say which
example and I will add it.

'''
pip install larder
'''


Work-in-Progress
----------------

We're working on some alternatives to crude. Less complete, but simpler. 
The idea is maybe to one day refactor `crude.py` to start with simpler functions 
that implement the most frequent use cases, and then add layers that extend it's 
functionality, all the way to our current `crude.py` coverage.

- `wip/input_sourcing.py` — the `source_variables` decorator that resolves
	function arguments from a mall (useful for APIs and FastAPI endpoints).
- `wip/input_wire.py` — a small `input_wiring` helper that binds function
	parameters to values from a dependency-injector style mall.
    
**Resolve arguments from a mall using `source_variables`**

```python
from larder.wip.input_sourcing import source_variables, mock_mall

@source_variables(
		segments={'resolver': lambda v, sk, mall: mall[sk].get(v), 'store_key': 'segments'},
)
def embed_text(segments, embedder):
		return embedder(segments)

# mock_mall['segments']['greeting'] == 'hello' and mock_mall['embedders']['default'] exists
result = embed_text(segments='greeting', embedder=mock_mall['embedders']['default'])
```

Why useful: API endpoints receive simple keys (or direct values). The
decorator transparently resolves keys to Python objects before the handler
runs. Use cases include FastAPI endpoints, CLI wrappers, and simple webhooks.

**Simple input wiring (dependency-injector style)**

```python
from larder.wip.input_wire import input_wiring

class MockMall:
		store1 = lambda: {'x': 10}
		store2 = lambda: {'y': 5}

def add(a, b, c):
		return a + b * c

wrapped = input_wiring(add, global_param_to_store={'a':'store1', 'b':'store2'}, mall=MockMall())
res = wrapped('x', 'y', 2)  # resolves 'x' -> 10, 'y' -> 5
assert res == 20
```

Why useful: small bridge to dependency-injection style stores. Helpful when
tests or apps already expose stores via an object with callable attributes.
