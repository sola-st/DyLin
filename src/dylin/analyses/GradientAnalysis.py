from typing import Any, Callable, Dict, List, Optional, Tuple
import collections

from torch import Tensor
from .base_analysis import BaseDyLinAnalysis
from ..markings.obj_identifier import uniqueid, get_ref, add_cleanup_hook, save_uid

import tensorflow as tf
import torch.nn as nn
import torch


class GradientAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_name = "GradientAnalysis"
        # common clipping values are 1,3,5,8,10
        self.threshold = float(10.0)
        self.stored_torch_models: Dict[str, bool] = {}
        self.total_gradients_investigated = 0

        def cleanup_torch_model(uuid: str):
            """
            removes model uuid as soon as gc collects it
            """
            if not self.stored_torch_models.get(uuid) is None:
                del self.stored_torch_models[uuid]

        add_cleanup_hook(cleanup_torch_model)

    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict):
        # tensorflow
        # print(f"{self.analysis_name} pre_call {iid}")
        if "__func__" in dir(function) and function.__func__ == tf.optimizers.Optimizer.apply_gradients:
            if isinstance(pos_args[0], collections.abc.Iterator):
                # pos_args[0] can be a zip object, which is an Iterator. These objects
                # can only be used once and then return an emtpy list, to prevent that
                # we reassign pos_args with the extracted list in the end
                # pos_args = (l,)
                gradients = list(pos_args[0])
                pos_args[0] = gradients
            else:
                gradients = list(pos_args[0])

            for i in range(0, len(gradients)):
                self.total_gradients_investigated = self.total_gradients_investigated + 1
                # gradients[i] is a tuple where first element is gradient, second trainable variable
                grad: tf.Tensor = gradients[i][0]
                _min = tf.math.reduce_min(grad)
                _max = tf.math.reduce_max(grad)
                if _min <= -self.threshold or _max >= self.threshold:
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "M-28",
                        f"Gradient too high / low gradient has min {_min} max {_max} value",
                    )

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        val: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        # print(f"{self.analysis_name} post_call {iid}")
        if val is function:
            return
        # pytorch

        # nn.Module is the base class for all neural network modules
        # mirror the object and use it later to extract gradients
        if isinstance(val, nn.Module):
            uuid = save_uid(val)
            if self.stored_torch_models.get(uuid) is None:
                self.stored_torch_models[uniqueid(val)] = True
        else:
            # because Optimizer.step sometimes use annotations which hide actual method
            # we just hook every call to Optimizers
            _self = getattr(function, "__self__", lambda: None)
            if _self is not None and isinstance(_self, torch.optim.Optimizer):
                for model_uid in self.stored_torch_models:
                    ref = get_ref(model_uid)
                    if not ref is None:
                        model: nn.Module = ref()
                        # extract params, they include all params for the nn.Module instance
                        # e.g. weights, bias etc. we extract only the ones for which a gradient is available
                        params: List[nn.parameter.Parameter] = model.parameters()
                        grads: List[Optional[Tensor]] = [p.grad for p in params]
                        if len(grads) > 0:
                            self.total_gradients_investigated = self.total_gradients_investigated + 1
                        for grad in grads:
                            if not grad is None:
                                _max = torch.max(grad)
                                _min = torch.min(grad)
                                if _max >= self.threshold or _min <= -self.threshold:
                                    self.add_finding(
                                        iid,
                                        dyn_ast,
                                        "M-28",
                                        f"Gradient too high / low gradient has min {_min} max {_max} value",
                                    )
                                    return

    def end_execution(self) -> None:
        self.add_meta({"total_gradients_investigated": self.total_gradients_investigated})
        super().end_execution()
