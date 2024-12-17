import time

def rate_limit(func):
    def wrapper(*args, **kwargs):
        time.sleep(1)  # 100ms between calls
        return func(*args, **kwargs)
    return wrapper