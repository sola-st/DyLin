old_stdout = sys.stdout
# flush sys.stdout
sys.stdout.flush()

# close sys.stdout
sys.stdout.close() # DyLin warn

# reopen sys.stdout to prevent program from crashing
sys.stdout = io.TextIOWrapper(io.FileIO(1, 'w'), write_through=True)