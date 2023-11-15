import numpy as np
from multiprocessing import shared_memory


def store_in_shm(data):
    shm = shared_memory.SharedMemory(name='foo', create=True, size=data.nbytes)
    shmData = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shmData[:] = data[:]
    #there must always be at least one `SharedMemory` object open for it to not
    #  be destroyed on Windows, so we won't `shm.close()` inside the function,
    #  but rather after we're done with everything.
    return shm


def read_from_shm(shape, dtype):
    shm = shared_memory.SharedMemory(name='foo', create=False)
    shmData = np.ndarray(shape, dtype, buffer=shm.buf)
    print('From read_from_shm():', shmData)
    return shm, shmData #we need to keep a reference of shm both so we don't
                        #  segfault on shmData and so we can `close()` it.


if __name__ == '__main__':
    data = np.arange(10)
    shm1 = store_in_shm(data)
    #This is where the *Windows* previously reclaimed the memory resulting in a
    #  FileNotFoundError because the tempory mmap'ed file had been released.
    shm2, shmData = read_from_shm(data.shape, data.dtype)
    print('From __main__:', shmData)
    shm1.close()
    shm2.close()
    #on windows "unlink" happens automatically here whether you call `unlink()` or not.
    shm2.unlink() #either shm1 or shm2
