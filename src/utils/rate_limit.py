'''
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_rpm=60, max_tokens_per_min=60000, initial_backoff=0.1, backoff_factor=2):
        self.max_rpm = max_rpm
        self.max_tokens_per_min = max_tokens_per_min
        self.token_usage = deque()
        self.backoff_time = initial_backoff
        self.backoff_factor = backoff_factor

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            while True:
                current_time = time.time()

                # Remove old token usage
                while self.token_usage and current_time - self.token_usage[0][0] > 60:
                    self.token_usage.popleft()

                # Calculate remaining tokens and delay
                used_tokens = sum(t[1] for t in self.token_usage)
                remaining_tokens = self.max_tokens_per_min - used_tokens
                estimated_tokens = kwargs.get('estimated_tokens', 1)

                if len(self.token_usage) < self.max_rpm and remaining_tokens > estimated_tokens:
                    break

                # Exponential backoff
                time.sleep(self.backoff_time)
                self.backoff_time *= self.backoff_factor

            self.token_usage.append((time.time(), estimated_tokens))
            self.backoff_time = max(self.backoff_time / self.backoff_factor, 0.1)
            return func(*args, **kwargs)

        return wrapper

@RateLimiter(max_rpm=60, max_tokens_per_min=60000)
def rate_limit(func):
    return func
'''

import time
from collections import deque

class RateLimiter:
    def __init__(self, 
                 max_rpm=2000,         
                 max_tokens_per_min=4000000,  
                 initial_backoff=0.1, 
                 backoff_factor=1.2):
        self.max_rpm = max_rpm
        self.max_tokens_per_min = max_tokens_per_min
        self.token_usage = deque()
        self.backoff_time = initial_backoff
        self.backoff_factor = backoff_factor

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            estimated_tokens = kwargs.get('estimated_tokens', 1)
            while True:
                current_time = time.time()
                # Remove old token usage entries older than 60 seconds
                while self.token_usage and current_time - self.token_usage[0][0] > 60:
                    self.token_usage.popleft()

                used_tokens = sum(t[1] for t in self.token_usage)
                remaining_tokens = self.max_tokens_per_min - used_tokens

                # If we have capacity for at least one more request and enough tokens:
                if len(self.token_usage) < self.max_rpm and remaining_tokens >= estimated_tokens:
                    break

                # If not, apply minimal delay before re-checking
                time.sleep(self.backoff_time)

            # Record the usage
            self.token_usage.append((time.time(), estimated_tokens))
            return func(*args, **kwargs)

        return wrapper

@RateLimiter(max_rpm=2000, max_tokens_per_min=4000000, initial_backoff=0.1, backoff_factor=1)
def rate_limit(func):
    return func
