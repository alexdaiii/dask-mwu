import numpy as np
import dask.array as da


def np_shapes():
    arr = np.array([0, 1, 0, 1, 0])

    print(arr.shape)


def dask_chunking():
    base_arr = np.array(
        [
            [1.5, 2.7, 3.5, 4.02],
            [5.21, 6.1, 7.35, 8.12],
            [9.34, 10.134, 11.26, 12.82],
            [13.6578, 14.4534, 15.415, 16.6456],
            [17.262365, 18.254636, 19.156462, 20.45254],
            [21.7452345, 22.942125, 23.0657, 24.4135],
        ]
    )

    chunks = {
        "along_col": (-1, 2),
        "along_row": (2, -1),
        "any": (2, 2),
    }

    def print_shape(arrMap: da.Array):
        print("Shape", arrMap.shape)

        # convert to integers
        return arrMap.astype(np.int64)

    for k, v in chunks.items():
        print(f"Chunking along {k} with {v} chunks")
        # Set meta using a small array with the same dtype
        meta = np.array([], dtype=base_arr.dtype)

        arr = da.from_array(base_arr, chunks=v, meta=meta)
        print("Chunks", arr.chunks)
        print("Chunksize", arr.chunksize)

        ret_val = arr.map_blocks(
            print_shape,
            meta=np.array([], dtype=np.int64),
        )

        print(ret_val._meta, ret_val.dtype)
        print(ret_val.compute())


if __name__ == "__main__":
    np_shapes()
    dask_chunking()
