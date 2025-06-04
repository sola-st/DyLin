# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    This specification ensures that canvas widgets are added only to the CanvasFrame's designated canvas
    source: https://www.nltk.org/api/nltk.draw.util.html#nltk.draw.util.CanvasFrame.add_widget
"""


class CreateWidgetOnSameFrameCanvas(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "CreateWidgetOnSameFrameCanvas"

    @only(patterns=["add_widget"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["nltk.draw.util.CanvasFrame"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, '__class__'):
            cls = function.__self__.__class__
            class_name = cls.__module__ + "." + cls.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            args = pos_args
            kwargs = kw_args

            canvasFrame = function.__self__  # Updated to use the self object
            canvasWidget = None

            if len(args) > 1:
                canvasWidget = args[1]
            else:
                canvasWidget = kwargs['canvaswidget']

            fCanvas = canvasFrame.canvas()
            wCanvas = canvasWidget.canvas()

            # TODO: Do we need to recursively check the children of the CanvasWidget?
            # Logically, it makes sense, but docs don't mention it directly.

            if wCanvas.winfo_id() != fCanvas.winfo_id():

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-5",
                    f"CanvasWidget must be created on the same canvas as the CanvasFrame it is being added to at {dyn_ast}."
                )
# =========================================================================
