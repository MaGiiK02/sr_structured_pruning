
# See https://github.com/MingSun-Tse/Regularization-Pruning/blob/master/utils.py#L286
class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr
        
        
class PresetLRScheduler(object):
    """
    Custom Implementation of 
    See: https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Srinivas_Cyclical_Pruning_for_Sparse_Neural_Networks_CVPRW_2022_paper.pdf
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr