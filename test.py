import numpy as np
from multiprocessing import shared_memory


def store_in_shm(data):
    shm = shared_memory.SharedMemory(name='foo', create=True, size=data.nbytes)
    shmData = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shmData[:] = data[:]
    return shm


def read_from_shm(shape, dtype):
    shm = shared_memory.SharedMemory(name='foo', create=False)
    shmData = np.ndarray(shape, dtype, buffer=shm.buf)
    print('From read_from_shm():', shmData)
    return shm, shmData  # we need to keep a reference of shm both so we don't
    #  segfault on shmData and so we can `close()` it.


if __name__ == '__main__':
    data = np.arange(10)
    shm = store_in_shm(data)
    shmData = read_from_shm(data.shape, data.dtype)
    print('From __main__:', shmData)    # no seg fault if we comment this line
    shm.close()
    shm.unlink()
