import pytest

from larder.crude import store_on_output


class SimpleDictStore(dict):
    pass


def test_store_on_output_sync():
    output_store = {}

    @store_on_output("result", store=output_store)
    def add(a, b):
        return a + b

    result = add(2, 3)
    assert result == 5
    assert "result" in output_store
    assert output_store["result"] == 5


@pytest.mark.asyncio
async def test_store_on_output_async():
    output_store = {}

    @store_on_output("result", store=output_store)
    async def add_async(a, b):
        return a + b

    result = await add_async(5, 6)
    assert result == 11
    assert "result" in output_store
    assert output_store["result"] == 11


def test_auto_namer_output_store():
    output_store = {}

    def square(x):
        return x * x

    auto_namer = lambda *, arguments, output: f"square_{arguments['x']}"
    wrapped_square = store_on_output(store=output_store, auto_namer=auto_namer)(square)
    result = wrapped_square(4)
    assert result == 16
    assert "square_4" in output_store
    assert output_store["square_4"] == 16


def test_multi_value_output_store():
    output_store = {}

    def range_list(n):
        return list(range(n))

    auto_namer = lambda *, arguments, output: f"r_{arguments['n']}_{output}"
    wrapped_range = store_on_output(
        store=output_store,
        auto_namer=auto_namer,
        store_multi_values=True,
        save_name_param=None,
    )(range_list)
    result = wrapped_range(3)
    assert result == [0, 1, 2]
    for item in result:
        key = f"r_3_{item}"
        assert key in output_store
        assert output_store[key] == item


def test_simple_output_store():
    output_store = {}

    def add(a, b):
        return a + b

    wrapped = store_on_output("addition_test", store=output_store)(add)
    res = wrapped(2, 3)
    assert res == 5
    assert "addition_test" in output_store
    assert output_store["addition_test"] == 5
