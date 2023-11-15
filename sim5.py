import concurrent
import time
from multiprocessing import shared_memory
import numpy as np
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

NUM_WORKERS = multiprocessing.cpu_count()
np.random.seed(42)
ARRAY_SIZE = int(2e8)
ARRAY_SHAPE = (ARRAY_SIZE,)
NP_SHARED_NAME = 'npshared'
NP_DATA_TYPE = np.float64
data = np.random.random(ARRAY_SIZE)


def create_shared_memory_nparray(data):
    d_size = np.dtype(NP_DATA_TYPE).itemsize * np.prod(ARRAY_SHAPE)

    shm = shared_memory.SharedMemory(create=True, size=d_size, name=NP_SHARED_NAME)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
    dst[:] = data[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()  # Free and release the shared memory block


def np_sum(name, start, stop):
    # not mandatory to init it here, just for demostration purposes.
    shm = shared_memory.SharedMemory(name=name)
    np_array = np.ndarray(ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
    return np.sum(np_array[start:stop])


def benchmark():
    chunk_size = int(ARRAY_SIZE / NUM_WORKERS)
    futures = []
    ts = time.time_ns()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(0, NUM_WORKERS):
            start = i + chunk_size if i == 0 else 0
            futures.append(executor.submit(np_sum, NP_SHARED_NAME, start, i + chunk_size))
    futures, _ = concurrent.futures.wait(futures)
    return (time.time_ns() - ts) / 1_000_000


if __name__ == '__main__':
    shm = create_shared_memory_nparray(data)
    time_spent = benchmark()
    print(f'spent: {time_spent}')
    release_shared(NP_SHARED_NAME)
