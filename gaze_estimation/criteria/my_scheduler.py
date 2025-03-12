



class CustomStepLR:
    def __init__(self, optimizer, step_size, gamma=0.8, step_frequency=100):
        """
        Initializes the custom learning rate scheduler.

        Parameters:
        - optimizer: The optimizer for which to adjust the learning rate.
        - step_size: The number of steps after which the learning rate is decayed.
        - gamma: The factor by which the learning rate will be multiplied.
        - step_frequency: The frequency (in number of steps) at which to check and apply the learning rate decay.
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_frequency = step_frequency
        self.step_count = 0

    def step(self):
        """
        Update the learning rate based on the current step count.
        """
        self.step_count += 1
        if self.step_count % self.step_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma

    def get_lr(self):
        """
        Returns the current learning rate.
        """
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
