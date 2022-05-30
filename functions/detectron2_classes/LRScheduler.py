from detectron2.engine.train_loop import HookBase
from collections import Counter

class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    Modified to work with decrease lr on plateau
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.last_val_loss = None

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break
                    
    def before_train(self):
        self.trainer.val_loss = None

    def after_step(self):
        if self.trainer.cfg.SOLVER.LR_SCHEDULER_NAME == "WarmupValLossLR":
            curr_val_loss = self.trainer.val_loss
            if self.trainer.val_loss:
                if curr_val_loss != self.last_val_loss:
                    # A new loss has been calculated 
                    # (obviosly small change new loss will be same as old)
                    lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
                    self.trainer.storage.put_scalar(
                        "lr", lr, smoothing_hint=False)
                    self._scheduler.step(metrics=curr_val_loss)
                    self.last_val_loss = curr_val_loss
        else:
            lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
            self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
            self._scheduler.step()