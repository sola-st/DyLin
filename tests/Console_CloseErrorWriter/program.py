# import sys
# import io

# sys.stderr.flush()
# sys.stderr.close() # DyLin warn
# # reopen stderr to prevent program from crash
# sys.stderr = io.TextIOWrapper(io.FileIO(2, 'w'), write_through=True)