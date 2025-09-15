import pytest
from typing import (
    Callable,
    Any,
    Dict,
    List,
    Tuple,
    get_args,
    Sequence,
    NewType,
    Iterable,
    Literal,
)
from functools import partial
from collections.abc import MutableMapping

from dol import Pipe

from larder.dog import DOG, ADOG
from au.base import ComputationStatus
import time

Segment = NewType("Segment", str)
Embedding = NewType("Embedding", Sequence[float])
PlanarVector = Tuple[float, float]
ClusterIndex = NewType("ClusterIndex", int)

Segments = Iterable[Segment]
Embeddings = Iterable[Embedding]
PlanarVectors = Iterable[PlanarVector]
ClusterIndices = Iterable[ClusterIndex]

Embedder = Callable[[Segments], Embeddings]
Planarizer = Callable[[Embeddings], PlanarVectors]
Clusterer = Callable[[Embeddings], ClusterIndices]

vectorize = lambda func: Pipe(partial(map, func), list)

operation_signatures = {
    "embedder": Embedder,
    "planarizer": Planarizer,
    "clusterer": Clusterer,
}

data_stores = {
    "segments": {
        "type": Segments,
        "store": {
            "segments_1": ["segment1", "segment2", "segment3"],
            "segments_2": ["segment4", "segment5"],
        },
    },
    "embeddings": {"type": Embeddings, "store": dict()},
    "planar_vectors": {"type": PlanarVectors, "store": dict()},
    "cluster_indices": {"type": ClusterIndices, "store": dict()},
}

operation_implementations = {
    "embedder": {
        "constant": lambda segments: vectorize(lambda s: [1, 2, 3])(segments),
        "segment_based": lambda segments: vectorize(lambda s: [len(s), 0.5, 0.5])(
            segments
        ),
    },
    "planarizer": {
        "constant": lambda embeddings: vectorize(lambda e: (e[0], e[1]))(embeddings),
        "embedding_based": lambda embeddings: vectorize(
            lambda e: [e[0] * 0.5, e[1] * 0.5]
        )(embeddings),
    },
    "clusterer": {
        "kmeans": lambda embeddings: [0, 1]
        * (len(embeddings) // 2 + len(embeddings) % 2),
        "dbscan": lambda embeddings: [-1] * len(embeddings),
    },
}

import copy

data_stores_orig = copy.deepcopy(data_stores)


def get_test_stores_and_ops(stores_type: Literal["ram", "local"] = "ram"):
    if stores_type == "ram":
        import copy

        data_stores = copy.deepcopy(data_stores_orig)
        op_impls = operation_implementations
    else:
        from imbed.imbed_project import get_mall

        mall = get_mall("dog_tests", get_project_mall=stores_type)
        data_stores = {
            "segments": {"type": Segments, "store": mall["segments"]},
            "embeddings": {"type": Embeddings, "store": mall["embeddings"]},
            "planar_vectors": {
                "type": PlanarVectors,
                "store": mall["planar_embeddings"],
            },
            "cluster_indices": {"type": ClusterIndices, "store": mall["clusters"]},
        }
        op_impls = operation_implementations
        for name, store in data_stores.items():
            s = store["store"]
            rootdir = getattr(s, "rootdir", None)
            print(f"  {name}: {rootdir}")
        for store in data_stores.values():
            keys = list(store["store"].keys())
            for k in keys:
                try:
                    del store["store"][k]
                except Exception:
                    pass
        data_stores["segments"]["store"].update(
            {
                "segments_1": ["segment1", "segment2", "segment3"],
                "segments_2": ["segment4", "segment5"],
            }
        )
    return data_stores, op_impls


def test_dog_operations(stores_type: Literal["ram", "local"] = "ram"):
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    dog = DOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
    )
    expected_data_stores = [
        "segments",
        "embeddings",
        "planar_vectors",
        "cluster_indices",
    ]
    assert sorted(list(dog.data_stores.keys())) == sorted(expected_data_stores)
    assert "segments_1" in dog.data_stores["segments"]
    dog.data_stores["segments"]["segments_3"] = ["segment6", "segment7"]
    assert "segments_3" in dog.data_stores["segments"]
    dog.data_stores["segments"]["segments_1"] = [
        "updated_segment_A",
        "updated_segment_B",
    ]
    assert dog.data_stores["segments"]["segments_1"] == [
        "updated_segment_A",
        "updated_segment_B",
    ]
    del dog.data_stores["segments"]["segments_3"]
    assert "segments_3" not in dog.data_stores["segments"]
    segments_for_embedding = dog.data_stores["segments"]["segments_1"]
    output_store_key_embed, output_val_key_embed = dog.call(
        dog.operation_implementations["embedder"]["constant"], segments_for_embedding
    )
    assert output_store_key_embed == "embeddings"
    retrieved_embeddings = dog.data_stores[output_store_key_embed][output_val_key_embed]
    assert retrieved_embeddings == [[1, 2, 3], [1, 2, 3]]
    output_store_key_planar, output_val_key_planar = dog.call(
        dog.operation_implementations["planarizer"]["embedding_based"],
        retrieved_embeddings,
    )
    assert output_store_key_planar == "planar_vectors"
    retrieved_planar_vectors = dog.data_stores[output_store_key_planar][
        output_val_key_planar
    ]
    assert retrieved_planar_vectors == [[0.5, 1.0], [0.5, 1.0]]
    output_store_key_cluster, output_val_key_cluster = dog.call(
        dog.operation_implementations["clusterer"]["kmeans"], retrieved_embeddings
    )
    assert output_store_key_cluster == "cluster_indices"
    retrieved_cluster_indices = dog.data_stores[output_store_key_cluster][
        output_val_key_cluster
    ]
    assert retrieved_cluster_indices == [0, 1]


# Test ADOG operations
def test_adog_operations(stores_type: Literal["ram", "local"] = "ram"):
    """
    Test the ADOG async operation graph. This mirrors the DOG test but checks async result storage.
    """
    # Use a fresh data_stores dict for each test
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    adog = ADOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
    )

    # CRUD: Add and update segments
    adog.data_stores["segments"]["segments_3"] = ["segment6", "segment7"]
    adog.data_stores["segments"]["segments_1"] = [
        "updated_segment_A",
        "updated_segment_B",
    ]
    del adog.data_stores["segments"]["segments_3"]

    # --- Async Operation Execution ---
    segments_for_embedding = adog.data_stores["segments"]["segments_1"]
    output_store_key_embed, output_val_key_embed = adog.call(
        adog.operation_implementations["embedder"]["constant"], segments_for_embedding
    )
    # Wait for async result to appear in the store
    for _ in range(50):  # up to 5 seconds
        if output_val_key_embed in adog.data_stores[output_store_key_embed]:
            computation_result = adog.data_stores[output_store_key_embed][
                output_val_key_embed
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
            elif computation_result.status == ComputationStatus.FAILED:
                raise RuntimeError(
                    f"ADOG embedder computation failed: {computation_result.error}"
                )
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG embedder async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_embed][output_val_key_embed]
    retrieved_embeddings = computation_result.value
    assert retrieved_embeddings == [[1, 2, 3], [1, 2, 3]]

    # Planarizer async
    output_store_key_planar, output_val_key_planar = adog.call(
        adog.operation_implementations["planarizer"]["embedding_based"],
        retrieved_embeddings,
    )
    for _ in range(50):
        if output_val_key_planar in adog.data_stores[output_store_key_planar]:
            computation_result = adog.data_stores[output_store_key_planar][
                output_val_key_planar
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG planarizer async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_planar][
        output_val_key_planar
    ]
    retrieved_planar_vectors = computation_result.value
    assert retrieved_planar_vectors == [[0.5, 1.0], [0.5, 1.0]]

    # Clusterer async
    output_store_key_cluster, output_val_key_cluster = adog.call(
        adog.operation_implementations["clusterer"]["kmeans"],
        retrieved_embeddings,
    )
    for _ in range(50):
        if output_val_key_cluster in adog.data_stores[output_store_key_cluster]:
            computation_result = adog.data_stores[output_store_key_cluster][
                output_val_key_cluster
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG clusterer async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_cluster][
        output_val_key_cluster
    ]
    retrieved_cluster_indices = computation_result.value
    assert retrieved_cluster_indices == [0, 1]

    print("\n--- All ADOG async operations tested successfully! ---")


def test_dog_sourced_operation(stores_type: Literal["ram", "local"] = "ram"):
    """
    Test DOG with sourced_argnames: when an argument name is in sourced_argnames,
    and the value passed is a key, the value is fetched from the corresponding store.
    """
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    sourced_argnames = {"segments": "segments", "embeddings": "embeddings"}
    dog = DOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
        sourced_argnames=sourced_argnames,
    )
    # Store a new segments list under a key
    dog.data_stores["segments"]["my_segments"] = ["a", "b", "c"]
    # Call embedder with a key, not the value
    output_store, output_key = dog.call(
        dog.operation_implementations["embedder"]["constant"], segments="my_segments"
    )
    result = dog.data_stores[output_store][output_key]
    assert result == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # Now test chaining: planarizer with embeddings key
    dog.data_stores["embeddings"]["my_embeds"] = [[1, 2, 3], [4, 5, 6]]
    output_store2, output_key2 = dog.call(
        dog.operation_implementations["planarizer"]["constant"], embeddings="my_embeds"
    )
    result2 = dog.data_stores[output_store2][output_key2]
    assert list(map(list, result2)) == [[1, 2], [4, 5]]
    print("test_dog_sourced_operation passed.")
