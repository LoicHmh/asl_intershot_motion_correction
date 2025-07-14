import time
import contextlib


@contextlib.contextmanager
def clock(msg: str = "", cpu_time=False):
    print(f"{msg} start ...")
    if cpu_time:
        start_time = time.process_time()  
        yield  
        end_time = time.process_time()  
        print(f"{msg} CPU time used: {end_time - start_time:.2f} s.")

    else:
        start_time = time.time()
        yield  
        end_time = time.time()
        print(f"{msg} time used: {end_time - start_time:.2f} s.")


    


# Define a decorator that uses the context manager
def clock_func(msg=''):                     # decorator_factory
    def clock_func_(func):                  # decorator
        def wrapper(*args, **kwargs):       # wrapper
            with clock(msg):                # function
                return func(*args, **kwargs)
        return wrapper
    return clock_func_


if __name__ == "__main__":
    # Example usage: 1
    @clock_func(msg='try')
    def example_function(n):
        sum = 0
        for i in range(n):
            sum += i ** 2
        return sum
    # Call the function to see the CPU time
    result = example_function(1000000)

    # Example usage: 2
    with clock("Example function"):
        result = example_function(1000000)

    with clock("Example function"):
        sum = 0
        for i in range(1000000):
            sum += i ** 2