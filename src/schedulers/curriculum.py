import math
import numpy as np

class CurriculumScheduler:
    def __init__(self, s0=10, s1=1280, k_max=1000):
        self.s0 = s0
        self.s1 = s1
        self.k_max = k_max
    
class ExpCurriculumScheduler(CurriculumScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k_prime = math.floor(
            self.k_max
            / (math.log2(math.floor(self.s1/self.s0)) + 1)
        )

    def __call__(self, k):
        res = np.min([
            self.s0 * 2**(math.floor(k/self.k_prime)),
            self.s1
        ])
        return int(res)
    
class CurriculumSchedulerPretraining(ExpCurriculumScheduler):
    def __init__(self, s0=10, s1=1280, k_max=1000, pre_ratio=1/3):
        self.pre_steps = int(k_max*pre_ratio)
        k_max = k_max - self.pre_steps
        super().__init__(s0=s0, s1=s1, k_max=k_max)
    
    def __call__(self, k):
        if k <= self.pre_steps:
            return 2
        else:
            return super().__call__(k-self.pre_steps)
    
class LinearCurriculumScheduler(CurriculumScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = (self.s1 - self.s0) / self.k_max

    def __call__(self, k):
        res = min(self.s0 + k * self.delta, self.s1)
        return int(res)
    
class BaseCurriculumScheduler(CurriculumScheduler):
    def __call__(self, k):
        summation = (k/self.k_max)*((self.s1+1)**2 - self.s0**2) + self.s0**2
        res = math.ceil(math.sqrt(summation)-1) + 1
        return res
    
class ConstantScheduler(CurriculumScheduler):
    def __call__(self, k):
        return self.s0