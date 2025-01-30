import torch

from tsl.ops.imputation import sample_mask

class PointStrategy:

    def get_mask(self, batch):
        shape = batch.x.shape
        missing_rate = torch.rand(shape[0], 1, 1, 1, device=batch.x.device)
        return torch.rand_like(batch.x) < missing_rate
    
    def __str__(self):
        return 'PointStrategy'
    
class BlockStrategy:
    def __init__(self, p=0.15, p_noise=0.05, min_seq=None, max_seq=24):
        self.p = p
        self.p_noise = p_noise
        self.max_seq = max_seq
        self.min_seq = min_seq if min_seq is not None else max_seq//2

    def get_mask(self, batch):
        shape = batch.x.shape
        device = batch.x.device

        p_values = torch.rand(shape[0], device=device) * self.p

        custom_mask = torch.stack([
            torch.tensor(
                sample_mask(
                    shape[1:],
                    p.item(),
                    self.p_noise,
                    self.min_seq,
                    self.max_seq,
                    verbose=False
                )
            , device=device) for p in p_values
        ])

        return custom_mask
    
    def __str__(self):
        return f'BlockStrategy: p={self.p}, p_noise={self.p_noise}, min_seq={self.min_seq}, max_seq={self.max_seq}'
    
class HistoryStrategy:
    def __init__(self, hist_patterns):
        self.hist_patterns = hist_patterns
        self.num_patterns = len(hist_patterns)
        
    def check_device(self, device):
        if self.hist_patterns.device != device:
            self.hist_patterns = self.hist_patterns.to(device)

    def get_mask(self, batch):
        B = batch.x.shape[0]
        device = batch.x.device
        self.check_device(device)

        index = torch.randint(self.num_patterns, (B,), device=device)
        patterns = self.hist_patterns[index].to(device)
        return patterns
    
    def __str__(self):
        return f'HistoryStrategy (num_patterns={self.num_patterns})'
    

class MissingPatternGenerator:
    def __init__(self, strategy1='point', strategy2=None, hist_patterns=None, seq_len=None):
        self.strategy1 = self.get_class_strategy(strategy1, hist_patterns, seq_len)
        self.strategy2 = self.get_class_strategy(strategy2, hist_patterns, seq_len)

        print(f'Strategy 1: {str(self.strategy1)}')
        print(f'Strategy 2: {str(self.strategy2)}')

    def get_class_strategy(self, strategy, hist_patterns, seq_len=None):
        if strategy == 'point':
            return PointStrategy()
        elif strategy == 'block':
            return BlockStrategy(max_seq=seq_len)
        elif strategy == 'historical':
            return HistoryStrategy(hist_patterns)
        else:
            return None
        
    def update_mask(self, batch):
        custom_mask = self.strategy1.get_mask(batch)
        new_mask = batch.mask & custom_mask

        if self.strategy2 is not None:
            B = batch.x.shape[0]
            custom_mask2 = self.strategy2.get_mask(batch)
            new_mask2 = batch.mask & custom_mask2

            mask_selection = torch.rand(B, 1, 1, 1, device=batch.x.device) < 0.5
            new_mask = torch.where(mask_selection, new_mask, new_mask2)

        batch.mask = new_mask
        batch.input.x = batch.input.x * batch.mask
        
