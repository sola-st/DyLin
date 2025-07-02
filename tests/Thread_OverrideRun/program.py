import threading


def run():
    pass

# =============== OK ===============
my_thread = threading.Thread(target=run)
my_thread.start()
my_thread2 = threading.Thread(target=run)
my_thread2.start()


# =============== OK ===============
my_thread = threading.Thread(target=run)
my_thread.start()
my_thread2 = threading.Thread(target=run)
my_thread2.start()
my_thread2.join()


# =============== OK ===============
class MyThread(threading.Thread):
    def run(self):
        pass

my_thread = MyThread()
my_thread.start()


# ============= NOT OK =============
# creating a thread without changing the target
my_thread = threading.Thread()
my_thread.start() # DyLin warn


# ============= NOT OK =============
# creating a thread without changing the target
my_thread = threading.Thread()
my_thread.start() # DyLin warn
my_thread.join()

