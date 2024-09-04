from .base_analysis import BaseDyLinAnalysis


class ItemInListAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "ItemInListAnalysis"
        self.threshold = 100

    def _in(self, dyn_ast, iid, left, right, result):
        if type(right) == list and len(right) > self.threshold:
            self.add_finding(
                iid,
                dyn_ast,
                "PC-05",
                f"Searching for an item ({left}) in a long list is not efficient. Consider using a set.",
            )
