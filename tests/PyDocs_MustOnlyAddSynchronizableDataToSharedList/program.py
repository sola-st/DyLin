from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Manager
import socket

if __name__ == "__main__":
    with Manager() as manager:
        shared_list = manager.list()

        shared_list.append(42) # OK

        # append a shared list
        new_shared_list = manager.list()
        shared_list.append(new_shared_list) # OK

        shared_dict = manager.dict()
        shared_list.append(shared_dict) # OK

        raw_dict = {"key": "value"}
        shared_list.append(raw_dict) # DyLin warn
        assert shared_list[-1] is not raw_dict  # dict constructed on the other side, but it's not the same dict.

        raw_list = [1, 2, 3]
        shared_list.append(raw_list) # DyLin warn
        assert shared_list[-1] is not raw_list  # list constructed on the other side, but it's not the same list.

        s = socket.socket()
        shared_list.append(s) # DyLin warn
        assert shared_list[-1] is not s  # Socket constructed on the other side, but it's not the same socket.
        s.close()
        
        shm = SharedMemory(create=True, size=10)
        shared_list.append(shm) # DyLin warn
        assert shared_list[-1] is not shm  # Shared memory constructed on the other side, but it's not the same shared memory.
        shm.close()
