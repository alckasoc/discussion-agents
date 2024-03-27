"""Unit tests for general util functions."""

from discussion_agents.utils.general import shuffle_chunk_list


def test_shuffle_chunk_list() -> None:
    """Test shuffle_chunk_list."""
    lst = list(range(10)) + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunked_lst = shuffle_chunk_list(lst, k=3)
    gt_chunked_lst = [
        [7, 1, "H"],
        ["F", "E", 4],
        ["N", 3, "L"],
        ["T", "X", 2],
        ["I", "D", "Y"],
        [0, "O", 6],
        ["Z", "G", "K"],
        [8, "S", "B"],
        ["A", "V", "Q"],
        ["M", "P", "U"],
        ["J", "W", "R"],
        [5, "C", 9],
    ]
    assert chunked_lst == gt_chunked_lst
