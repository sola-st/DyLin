# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict
import re


"""
    RegexpTokenizer pattern must not contain capturing parentheses
    src: https://www.nltk.org/api/nltk.tokenize.regexp.html
"""


def contains_capturing_groups(pattern):
    regex = re.compile(pattern)

    if regex.groups > 0:
        # Further check to distinguish capturing from non-capturing by examining the pattern
        # This involves checking all group occurrences in the pattern
        # We need to avoid matching escaped parentheses \( or \) and non-capturing groups (?: ...)
        non_capturing = re.finditer(r'\(\?[:=!]', pattern)
        non_capturing_indices = {match.start() for match in non_capturing}
        
        # Finding all parentheses that could start a group
        all_groups = re.finditer(r'\((?!\?)', pattern)
        for match in all_groups:
            if match.start() not in non_capturing_indices:
                return True  # Found at least one capturing group
        return False
    else:
        return False


class NLTK_RegexpTokenizerCapturingParentheses(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "NLTK_RegexpTokenizerCapturingParentheses"

    @only(patterns=["RegexpTokenizer"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["nltk.tokenize.regexp.RegexpTokenizer"]

        # Get the class name
        if hasattr(function, '__module__') and hasattr(function, '__name__'):
            class_name = function.__module__ + "." + function.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            pattern = None
            if kw_args.get('pattern'):
                pattern = kw_args['pattern']
            elif len(pos_args) > 1:
                pattern = pos_args[1]

            # Check if the regular expression is empty
            if pattern is not None and contains_capturing_groups(pattern):

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-9",
                    f"Must use non_capturing parentheses for RegexpTokenizer pattern at {dyn_ast}."
                )
# =========================================================================
