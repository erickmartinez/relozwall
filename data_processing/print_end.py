import time

if __name__ == '__main__':
    t0 = time.time()
    total_time = 0
    previous_time = 0
    while total_time <= 60:
        current_time = time.time()
        if (current_time - previous_time) >= 0.1:
            total_time = current_time - t0
            print(f"Time: {total_time:>6.3f}", end='\r', flush=True)
            previous_time = current_time
