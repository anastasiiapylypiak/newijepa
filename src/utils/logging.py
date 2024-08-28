import torch
import math

def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        # If CUDA is not available, use a simple CPU timer
        import time
        start_time = time.time()
        result = closure()
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return result, elapsed_time

class CSVLogger(object):
    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

class AverageMeter(object):
    """computes and stores the average, current value, min, max, and standard deviation"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0
        self.sum_of_squares = 0  # To store the sum of squares for std calculation
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Update the sum of squares
        self.sum_of_squares += n * (val ** 2)

        # Calculate variance (using Welford's method for numerical stability)
        if self.count > 1:
            mean_of_squares = self.sum_of_squares / self.count
            square_of_mean = self.avg ** 2
            variance = mean_of_squares - square_of_mean
            self.std = math.sqrt(variance)

    def __str__(self):
        return f'Avg: {self.avg:.4f}, Std: {self.std:.4f}, Min: {self.min}, Max: {self.max}, Count: {self.count}'


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats
