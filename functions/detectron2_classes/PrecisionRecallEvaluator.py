import itertools
import os

import numpy as np

from fvcore.common.file_io import PathManager

from detectron2.evaluation import COCOEvaluator
# from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import _create_text_labels

from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from pycocotools.cocoeval import Params
# from tabulate import tabulate

# import detectron2.utils.comm as comm
# from detectron2.config import CfgNode
# from detectron2.data import MetadataCatalog
# from detectron2.data.datasets.coco import convert_to_coco_json
# from detectron2.structures import Boxes, BoxMode, pairwise_iou
# from detectron2.utils.file_io import PathManager
# from detectron2.utils.logger import create_small_table

# from .evaluator import DatasetEvaluator
    
    
class MyCOCOeval(COCOeval):

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', min_iou=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        
        if min_iou:
            print(f"Using min iou threshold of {min_iou}.")
            new_iou_thresh = np.linspace(min_iou, 0.95, 
                                         int(np.round((0.95 - min_iou) / .05)) + 1, 
                                         endpoint=True)
            self.params.iouThrs = new_iou_thresh


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, 
                                  kpt_oks_sigmas=None, min_iou=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = MyCOCOeval(cocoGt=coco_gt, cocoDt=coco_dt, 
                           iouType=iou_type, min_iou=min_iou)

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

class MyCOCOEvaluator(COCOEvaluator):
     def __init__(self, dataset_name, cfg, distributed, output_dir=None, min_iou=None):

        super().__init__(dataset_name, cfg, distributed=distributed)
        self.min_iou = min_iou
        print('custom')
        if self.min_iou:
            print(f"Using custom min iou: {self.min_iou}")


class PrecisionRecallEvaluator(MyCOCOEvaluator):
    """ For generating PR curves. """
    
    
    def _eval_predictions(self, tasks, predictions, min_iou=None):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, 
                    kpt_oks_sigmas=self._kpt_oks_sigmas, min_iou=self.min_iou
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )

            self._results[task] = {'precision': coco_eval.eval['precision'],
                                   'params': coco_eval.eval['params'],
                                   'res': res,
                                   'scores': coco_eval.eval['scores'],
                                   'coco_eval': coco_eval.eval
                                  }
            
    
#     def _eval_predictions(self, tasks, predictions, min_iou=.5):
#         """
#         Evaluate predictions on the given tasks.
#         Fill self._results with the metrics of the tasks.
#         """
#         self._logger.info("Preparing results for COCO format ...")
#         coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

#         # unmap the category ids for COCO
#         if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
#             reverse_id_mapping = {
#                 v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
#             }
#             for result in coco_results:
#                 category_id = result["category_id"]
#                 assert (
#                     category_id in reverse_id_mapping
#                 ), "A prediction has category_id={}, which is not available in the dataset.".format(
#                     category_id
#                 )
#                 result["category_id"] = reverse_id_mapping[category_id]

#         if self._output_dir:
#             file_path = os.path.join(self._output_dir, "coco_instances_results.json")
#             self._logger.info("Saving results to {}".format(file_path))
#             with PathManager.open(file_path, "w") as f:
#                 f.write(json.dumps(coco_results))
#                 f.flush()

#         if not self._do_evaluation:
#             self._logger.info("Annotations are not available for evaluation.")
#             return

#         self._logger.info("Evaluating predictions ...")
#         for task in sorted(tasks):
#             coco_eval = (
#                 _evaluate_predictions_on_coco(
#                     self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
#                 )
#                 if len(coco_results) > 0
#                 else None  # cocoapi does not handle empty results very well
#             )

#             res = self._derive_coco_results(
#                 coco_eval, task, class_names=self._metadata.get("thing_classes")
#             )

#             self._results[task] = {'precision': coco_eval.eval['precision'],
#                                    'params': coco_eval.eval['params'],
#                                    'res': res,
#                                    'scores': coco_eval.eval['scores'],
#                                    'coco_eval': coco_eval.eval
#                                   }
            