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
    "PC-05": {
        "name": "ItemInList",
        "analysis": "dylin.analyses.ItemInListAnalysis.ItemInListAnalysis",
        "aliases": [],
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
    "SL-07": {
        "name": "Console_CloseErrorWriter",
        "analysis": "dylin.analyses.Console_CloseErrorWriter.Console_CloseErrorWriter",
        "aliases": ["B-2"],
    },
    "SL-08": {
        "name": "Console_CloseReader",
        "analysis": "dylin.analyses.Console_CloseReader.Console_CloseReader",
        "aliases": ["B-3"],
    },
    "SL-09": {
        "name": "Console_CloseWriter",
        "analysis": "dylin.analyses.Console_CloseWriter.Console_CloseWriter",
        "aliases": ["B-4"],
    },
    "SL-10": {
        "name": "PyDocs_MustOnlyAddSynchronizableDataToSharedList",
        "analysis": "dylin.analyses.PyDocs_MustOnlyAddSynchronizableDataToSharedList.PyDocs_MustOnlyAddSynchronizableDataToSharedList",
        "aliases": ["B-11"],
    },
    "SL-11": {
        "name": "RandomParams_NoPositives",
        "analysis": "dylin.analyses.RandomParams_NoPositives.RandomParams_NoPositives",
        "aliases": ["B-12"],
    },
    "SL-12": {
        "name": "RandomRandrange_MustNotUseKwargs",
        "analysis": "dylin.analyses.RandomRandrange_MustNotUseKwargs.RandomRandrange_MustNotUseKwargs",
        "aliases": ["B-13"],
    },
    "SL-13": {
        "name": "Thread_OverrideRun",
        "analysis": "dylin.analyses.Thread_OverrideRun.Thread_OverrideRun",
        "aliases": ["B-20"],
    },
    "CF-01": {
        "name": "WrongOperatorOverriding",
        "analysis": "dylin.analyses.ComparisonBehaviorAnalysis.ComparisonBehaviorAnalysis",
        "aliases": ["A-01", "A-03", "A-04"],
    },
    # "CF-02": {
    #     "name": "MutableDefaultArgs",
    #     "analysis": "dylin.analyses.MutableDefaultArgsAnalysis.MutableDefaultArgsAnalysis",
    #     "aliases": ["A-10"],
    # },
    "ML-01": {
        "name": "InconsistentPreprocessing",
        "analysis": "dylin.analyses.InconsistentPreprocessing.InconsistentPreprocessing",
        "aliases": ["M-23"],
    },
    "ML-02": {
        "name": "DataLeakage",
        "analysis": f"dylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis;config={here/'markings/configs/leaked_data.yml'}"
        + f"\ndylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis;config={here/'markings/configs/leak_preprocessing.yml'}",
        "aliases": ["M-24", "M-25"],
    },
    "ML-03": {
        "name": "NonFiniteValues",
        "analysis": "dylin.analyses.NonFinitesAnalysis.NonFinitesAnalysis"
        + "\ndylin.analyses.TensorflowNonFinitesAnalysis.TensorflowNonFinitesAnalysis",
        "aliases": ["M-26", "M-27", "M-32", "M-33"],
    },
    "ML-04": {
        "name": "GradientExplosion",
        "analysis": "dylin.analyses.GradientAnalysis.GradientAnalysis",
        "aliases": ["M-28"],
    },
    "TP-01": {
        "name": "CreateWidgetOnSameFrameCanvas",
        "analysis": "dylin.analyses.CreateWidgetOnSameFrameCanvas.CreateWidgetOnSameFrameCanvas",
        "aliases": ["B-5"],
    },
    "TP-02": {
        "name": "HostnamesTerminatesWithSlash",
        "analysis": "dylin.analyses.HostnamesTerminatesWithSlash.HostnamesTerminatesWithSlash",
        "aliases": ["B-6"],
    },
    "TP-03": {
        "name": "NLTK_regexp_span_tokenize",
        "analysis": "dylin.analyses.NLTK_regexp_span_tokenize.NLTK_regexp_span_tokenize",
        "aliases": ["B-8"],
    },
    "TP-04": {
        "name": "NLTK_RegexpTokenizerCapturingParentheses",
        "analysis": "dylin.analyses.NLTK_RegexpTokenizerCapturingParentheses.NLTK_RegexpTokenizerCapturingParentheses",
        "aliases": ["B-9"],
    },
    "TP-05": {
        "name": "Requests_DataMustOpenInBinary",
        "analysis": "dylin.analyses.Requests_DataMustOpenInBinary.Requests_DataMustOpenInBinary",
        "aliases": ["B-14"],
    },
    "TP-06": {
        "name": "Session_DataMustOpenInBinary",
        "analysis": "dylin.analyses.Session_DataMustOpenInBinary.Session_DataMustOpenInBinary",
        "aliases": ["B-15"],
    },
}


def select_checkers(include: str = "all", exclude: str = "none", output_dir: str = None) -> str:
    if include is None:
        include = "none"
    if exclude is None:
        exclude = "none"
    if include.lower() == "all" and exclude.lower() == "none":
        res = "\n".join([issue["analysis"] for _, issue in issue_codes.items()])
    elif include.lower() == "all":
        res = "\n".join(
            [
                issue["analysis"]
                for code, issue in issue_codes.items()
                if (code not in exclude and issue["name"] not in exclude)
            ]
        )
    elif exclude.lower() == "none":
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
