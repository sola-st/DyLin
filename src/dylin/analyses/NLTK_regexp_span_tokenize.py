# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    Regular expression passed to regexp_span_tokenize must not be empty
    src: https://www.nltk.org/api/nltk.tokenize.util.html
"""


class NLTK_regexp_span_tokenize(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "NLTK_regexp_span_tokenize"

    @only(patterns=["regexp_span_tokenize"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["nltk.tokenize.util"]

        # Get the class name
        if hasattr(function, '__module__'):
            class_name = function.__module__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            regexp = None
            if kw_args.get('regexp'):
                regexp = kw_args['regexp']
            elif len(pos_args) > 1:
                regexp = pos_args[1]

            # Check if the regular expression is empty
            if regexp == '':

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-8",
                    f"Regular expression must not be empty at {dyn_ast}."
                )
# =========================================================================
