from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
import numpy as np
import os

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, checkpointer=None):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        # For model checkpoint saving based on best loss
        self._checkpointer = checkpointer
        self.best_loss = np.inf
        self.last_filename = None
        
    def save_checkpoint_on_best_val(self):
        if self.best_loss > self.trainer.val_loss:
            if self.last_filename:
                last_file = os.path.join(
                    self._checkpointer.save_dir, 
                    self.last_filename + '.pth'
                )
                if os.path.exists(last_file):
                    os.remove(last_file)
            name = f"best-model-config-iter-{self.trainer.iter}-loss-{self.trainer.val_loss}"
            print(
                f"Saving new best model named {name} with val loss {self.trainer.val_loss}"
            )
            self._checkpointer.save(name)
            self.last_filename = name
            self.best_loss = self.trainer.val_loss
            
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.val_loss = mean_loss
        if self._checkpointer:
            self.save_checkpoint_on_best_val()

        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

        