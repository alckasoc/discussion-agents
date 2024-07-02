"""Main file for managing benchmark prompts/few-shot examples."""

from typing import Dict

from agential.cog.prompts.benchmark.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmark.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)


class Benchmarks:
    """Supported benchmarks."""

    QA = "qa"
    MATH = "math"
    CODE = "code"

    class qa:
        """qa benchmarks."""

        HOTPOTQA = "hotpotqa"
        FEVER = "fever"
        TRIVIAQA = "triviaqa"
        AMBIGNQ = "ambignq"

    class math:
        """math benchmarks."""

        GSM8K = "gsm8k"
        SVAMP = "svamp"
        TABMWP = "tabmwp"

    class code:
        """code benchmarks."""

        HUMANEVAL = "humaneval"
        MBPP = "mbpp"


class FewShotType:
    """Few-shot types."""

    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"


BENCHMARK_STRINGS = {
    Benchmarks.QA: {
        Benchmarks.qa.HOTPOTQA: {
            FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
            FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
            FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        },
        Benchmarks.qa.FEVER: {
            FewShotType.COT: FEVER_FEWSHOT_EXAMPLES_COT,
            FewShotType.DIRECT: FEVER_FEWSHOT_EXAMPLES_DIRECT,
            FewShotType.REACT: FEVER_FEWSHOT_EXAMPLES_REACT,
        },
        Benchmarks.qa.TRIVIAQA: {
            FewShotType.COT: TRIVIAQA_FEWSHOT_EXAMPLES_COT,
            FewShotType.DIRECT: TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
            FewShotType.REACT: TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
        },
        Benchmarks.qa.AMBIGNQ: {
            FewShotType.COT: AMBIGNQ_FEWSHOT_EXAMPLES_COT,
            FewShotType.DIRECT: AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
            FewShotType.REACT: AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
        },
    },
    Benchmarks.MATH: {
        Benchmarks.math.GSM8K: {
            FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: GSM8K_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: GSM8K_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.math.SVAMP: {
            FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: SVAMP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: SVAMP_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.math.TABMWP: {
            FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: TABMWP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: TABMWP_FEWSHOT_EXAMPLES_COT,
        },
    },
    Benchmarks.CODE: {
        Benchmarks.code.HUMANEVAL: {
            FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.code.MBPP: {
            FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: MBPP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: MBPP_FEWSHOT_EXAMPLES_COT,
        },
    },
}


def get_fewshot_examples(mode: Dict[str, str], fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark type and benchmark name.

    Available Benchmark Types and Names:
        - qa:
            - hotpotqa: Supports "cot", "direct", "react"
            - fever: Supports "cot", "direct", "react"
            - triviaqa: Supports "cot", "direct", "react"
            - ambignq: Supports "cot", "direct", "react"
        - math:
            - gsm8k: Supports "pot", "cot", "react"
            - svamp: Supports "pot", "cot", "react"
            - tabmwp: Supports "pot", "cot", "react"
        - code:
            - humaneval: Supports "pot", "cot", "react"
            - mbpp: Supports "pot", "cot", "react"

    Available Few-Shot Types:
        - "cot"
        - "direct"
        - "react"
        - "pot"

    Args:
        mode (dict): A dictionary with "benchmark type" as the key and "benchmark name" as the value.
        fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

    Returns:
        str: The few-shot examples corresponding to the given benchmark and type.
        If the benchmark or few-shot type is not found, returns a detailed error message.
    """
    benchmark_type, benchmark_name = list(mode.items())[0]
    if benchmark_type not in Benchmarks.__dict__:
        raise ValueError(f"Benchmark type '{benchmark_type}' not found.")

    if benchmark_name not in BENCHMARK_STRINGS[benchmark_type]:
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found in benchmark type '{benchmark_type}'."
        )

    examples = BENCHMARK_STRINGS[benchmark_type][benchmark_name].get(fewshot_type)
    if examples is None:
        raise ValueError(
            f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark_name}' of benchmark type '{benchmark_type}'."
        )

    return examples
