"""General utility functions."""

import math
import random

from typing import Any, List


def shuffle_chunk_list(lst: List[Any], k: int, seed: int = 42) -> List[List[Any]]:
    """Shuffles and divides the list into chunks, each with maximum length k.

    Args:
        lst (List[Any]): The list to be divided.
        k (int): The maximum length of each chunk.
        seed (int): The random seed. Defaults to 42.

    Returns:
        A list of chunks.

    Ref: https://github.com/LeapLabTHU/ExpeL.
    """
    random.seed(seed)

    lst = random.sample(lst, len(lst))

    if len(lst) <= k:
        return [lst]

    num_chunks = math.ceil(len(lst) / k)
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
