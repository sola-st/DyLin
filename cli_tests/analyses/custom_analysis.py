from dylin.analyses.base_analysis import BaseDyLinAnalysis


class CustomAnalysis(BaseDyLinAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_name = "CustomAnalysis"

    def write(self, dyn_ast, iid, old_vals, new_val):
        if new_val == 42:
            self.add_finding(iid, dyn_ast, "Custom", "New value is 42")