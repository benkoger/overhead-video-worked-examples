import torch
import os 

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader

from DetectionDatasetMapper import DetectionDatasetMapper
from LRScheduler import LRScheduler
from LossEvalHook import LossEvalHook

class DetectionTrainer(DefaultTrainer):
    """ Implements build_evaluator. 
    Only for coco.
    """
        
    @classmethod
    def build_lr_scheduler(self, cfg, optimizer):
        """
        Build a LR scheduler from config.
        Modified from default to allow reduce lr on plateau
        """
        name = cfg.SOLVER.LR_SCHEDULER_NAME
        if name == "WarmupMultiStepLR":
            return WarmupMultiStepLR(
                optimizer,
                cfg.SOLVER.STEPS,
                cfg.SOLVER.GAMMA,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        elif name == "WarmupCosineLR":
            return WarmupCosineLR(
                optimizer,
                cfg.SOLVER.MAX_ITER,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        elif name == "WarmupValLossLR":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2, 
                threshold=1e-4, 
                threshold_mode='rel', 
                cooldown=0,
                min_lr=0, 
                eps=1e-8, 
                verbose=True)
        else:
            raise ValueError("Unknown LR scheduler: {}".format(name))

        
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

        else:
            raise NotImplementedError(
                "Only coco evaluators are currently implemented." 
                "Change build_evaluator().")
            
            
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader` 
            but now with mapper.
        Overwrite it if you'd like a different data loader.
        """
        
        mapper = DetectionDatasetMapper(cfg, is_train=True)
        print(f'Train loader with custom mapper.')
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DetectionDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    
    def build_hooks(self):
        """
        Modified from default trainer build_hooks.
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        
        Modifyied to calculate validation loss and change learning 
        rate on validation loss.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        hooks_list = [
            hooks.IterationTimer(),
            LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if cfg.SOLVER.CHECKPOINT_PERIOD != 0:
            if comm.is_main_process():
                hooks_list.append(
                    hooks.PeriodicCheckpointer(
                        self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                    )
                )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        hooks_list.append(
            hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        
        if cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer = self.checkpointer
        else:
            checkpointer = None
        # hooks to caclulate the validation loss
        mapper = DetectionDatasetMapper(cfg, is_train=False, calc_val_loss=True)
        hooks_list.append(
            LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                   self.cfg,
                   self.cfg.DATASETS.TEST[0],
                   mapper
                ),
                checkpointer
            )
        )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            hooks_list.append(
                hooks.PeriodicWriter(self.build_writers(), period=20))
        
        print('hooks:', hooks_list)
        return hooks_list