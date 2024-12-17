import time

def rate_limit(func):
    last_call_time = time.time()
    def wrapper(*args, **kwargs):
        nonlocal last_call_time
        current_time = time.time()
        time_since_last_call = current_time - last_call_time
        min_delay = 60 / 15  # 15 RPM limit for gemini-1.5-flash
        if time_since_last_call < min_delay:
            time.sleep(min_delay - time_since_last_call)
        last_call_time = time.time()
        return func(*args, **kwargs)
    return wrapper