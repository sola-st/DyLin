from pathlib import Path
import fire

here = Path(__file__).parent.resolve()

issue_codes = {
    "PC-01": {
        "name": "InvalidFunctionComparison",
        "analysis": "dylin.analyses.InvalidComparisonAnalysis.InvalidComparisonAnalysis",
        "aliases": ["A-15"],
    },
    "PC-02": {
        "name": "RiskyFloatComparison",
        "analysis": "dylin.analyses.InvalidComparisonAnalysis.InvalidComparisonAnalysis",
        "aliases": ["A-12"],
    },
    "PC-03": {
        "name": "WrongTypeAdded",
        "analysis": "dylin.analyses.WrongTypeAddedAnalysis.WrongTypeAddedAnalysis",
        "aliases": ["A-11"],
    },
    "PC-04": {
        "name": "ChangeListWhileIterating",
        "analysis": "dylin.analyses.ChangeListWhileIterating.ChangeListWhileIterating",
        "aliases": ["A-22"],
    },
    "SL-01": {
        "name": "InPlaceSort",
        "analysis": "dylin.analyses.InPlaceSortAnalysis.InPlaceSortAnalysis",
        "aliases": ["A-09"],
    },
    "SL-02": {
        "name": "AnyAllMisuse",
        "analysis": "dylin.analyses.BuiltinAllAnalysis.BuiltinAllAnalysis",
        "aliases": ["A-21"],
    },
    "SL-03": {
        "name": "StringStrip",
        "analysis": "dylin.analyses.StringStripAnalysis.StringStripAnalysis",
        "aliases": ["A-19", "A-20"],
    },
    "SL-04": {
        "name": "StringConcat",
        "analysis": "dylin.analyses.StringConcatAnalysis.StringConcatAnalysis",
        "aliases": ["A-05"],
    },
    "SL-05": {
        "name": "InvalidTypeComparison",
        "analysis": "dylin.analyses.InvalidComparisonAnalysis.InvalidComparisonAnalysis",
        "aliases": ["A-13"],
    },
    "SL-06": {
        "name": "NondeterministicOrder",
        "analysis": f"dylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis;config={here/'markings/configs/forced_order.yml'}",
        "aliases": ["A-18"],
    },
    "CF-01": {
        "name": "WrongOperatorOverriding",
        "analysis": "dylin.analyses.ComparisonBehaviorAnalysis.ComparisonBehaviorAnalysis",
        "aliases": ["A-01", "A-03", "A-04"],
    },
    "CF-02": {
        "name": "MutableDefaultArgs",
        "analysis": "dylin.analyses.MutableDefaultArgsAnalysis.MutableDefaultArgsAnalysis",
        "aliases": ["A-10"],
    },
    "ML-01": {
        "name": "InconsistentPreprocessing",
        "analysis": "dylin.analyses.InconsistentPreprocessing.InconsistentPreprocessing",
        "aliases": ["M-23"],
    },
    "ML-02": {
        "name": "DataLeakage",
        "analysis": f"dylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis;config={here/'markings/configs/leaked_data.yml'}",
        "aliases": ["M-25"],
    },
    "ML-03": {
        "name": "NonFiniteValues",
        "analysis": "dylin.analyses.NonFinitesAnalysis.NonFinitesAnalysis",
        "aliases": ["M-32", "M-33"],
    },
    "ML-04": {
        "name": "GradientExplosion",
        "analysis": "dylin.analyses.GradientAnalysis.GradientAnalysis",
        "aliases": ["M-28"],
    },
}


def select_checkers(include: str = "all", exclude: str = "none", output_dir: str = None) -> str:
    if include == "all" and exclude == "none":
        res = "\n".join([issue["analysis"] for _, issue in issue_codes.items()])
    elif include == "all":
        res = "\n".join(
            [
                issue["analysis"]
                for code, issue in issue_codes.items()
                if (code not in exclude and issue["name"] not in exclude)
            ]
        )
    elif exclude == "none":
        res = "\n".join(
            [issue["analysis"] for code, issue in issue_codes.items() if (code in include or issue["name"] in include)]
        )
    else:
        res = "\n".join(
            [
                issue["analysis"]
                for code, issue in issue_codes.items()
                if (code in include or issue["name"] in include)
                and (code not in exclude and issue["name"] not in exclude)
            ]
        )
    if output_dir is not None:
        return "\n".join([f"{ana};output_dir={output_dir}" for ana in res.split("\n")])
    return res


if __name__ == "__main__":
    fire.Fire(select_checkers)
