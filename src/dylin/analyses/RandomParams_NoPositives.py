# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    random.lognormvariate(mu, sigma) -> mu can have any value, and sigma must be greater than zero.
    random.vonmisesvariate(mu, kappa) ->  kappa must be greater than or equal to zero.
"""


class RandomParams_NoPositives(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "RandomParams_NoPositives"

    @only(patterns=["lognormvariate", "vonmisesvariate"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["random"]

        # Get the class name
        if hasattr(function, '__module__'):
            class_name = function.__module__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Check if the parameters are correct
            violation = False
            if function.__name__ == "lognormvariate":
                sigma = None
                if kw_args.get('sigma'):
                    sigma = kw_args['sigma']
                elif len(pos_args) > 1:
                    sigma = pos_args[1]

                if sigma is not None and sigma <= 0:
                    violation = True

            else:  # Must be vonmisesvariate in this case
                kappa = None
                if kw_args.get('kappa'):
                    kappa = kw_args['kappa']
                elif len(pos_args) > 1:
                    kappa = pos_args[1]

                if kappa is not None and kappa < 0:
                    violation = True

            # If there is a violation, add a finding
            if violation:

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-12",
                    f"The call to method lognormvariate or vonmisesvariate in file at {dyn_ast} does not have the correct parameters. The sigma or kappa should always be positive."
                )
# =========================================================================
