"""
Test generator semantics and the capped text iterator pattern used by
scripts/tok_train.py to feed the BPE trainer.

The point of the iterator is that documents are produced lazily, one at a
time, so a 2B-character training run never materializes the corpus in memory.

python -m pytest tests/test_text_iterator.py -v
"""

import sys

import pytest


def counting_gen(log):
    """Appends to log as it runs so we can observe when the body executes."""
    log.append("A")
    yield 1
    log.append("B")
    yield 2
    log.append("C")


def text_iterator(batches, doc_cap, max_chars):
    """Mirror of the iterator in scripts/tok_train.py: flatten, crop, stop."""
    nchars = 0
    for batch in batches:
        for doc in batch:
            doc_text = doc
            if len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > max_chars:
                return


# -----------------------------------------------------------------------------
# generator semantics

def test_calling_a_generator_runs_no_code():
    log = []
    g = counting_gen(log)
    assert log == [] # the body has not started
    next(g)
    assert log == ["A"]


def test_yield_suspends_and_resumes():
    log = []
    g = counting_gen(log)
    assert next(g) == 1
    assert log == ["A"] # frozen at the first yield, "B" not reached yet
    assert next(g) == 2
    assert log == ["A", "B"]
    with pytest.raises(StopIteration):
        next(g)
    assert log == ["A", "B", "C"] # the tail after the last yield still runs


def test_generators_are_exhausted_once():
    log = []
    g = counting_gen(log)
    assert list(g) == [1, 2]
    assert list(g) == [] # no rewind: this is why tok_train.py builds text_iter once


# -----------------------------------------------------------------------------
# the tok_train.py iterator

def test_documents_are_cropped_to_doc_cap():
    docs = list(text_iterator([["a" * 20, "bb"]], doc_cap=5, max_chars=10**9))
    assert docs == ["aaaaa", "bb"]


def test_stops_once_the_character_budget_is_spent():
    batches = [["a" * 8, "bb"], ["c" * 6, "d" * 10], ["eeee"]]
    docs = list(text_iterator(batches, doc_cap=5, max_chars=12))
    # 5 + 2 + 5 = 12 is not yet over budget, the 4th document pushes it to 17
    assert docs == ["aaaaa", "bb", "ccccc", "ddddd"]
    # the check happens after the yield, so we overshoot by at most one doc_cap
    assert sum(len(d) for d in docs) <= 12 + 5


def test_later_batches_are_never_pulled_once_the_budget_is_spent():
    pulled = []

    def batches():
        for batch in [["a" * 10], ["b" * 10], ["c" * 10]]:
            pulled.append(batch[0][0])
            yield batch

    list(text_iterator(batches(), doc_cap=5, max_chars=6))
    # "a" and "b" together exceed the budget, so the third batch is never read
    assert pulled == ["a", "b"]


def test_production_and_consumption_interleave():
    order = []

    def batches():
        for i in range(4):
            order.append(f"produce{i}")
            yield [f"doc{i}"]

    for doc in text_iterator(batches(), doc_cap=10, max_chars=10**9):
        order.append(f"consume{doc[-1]}")

    assert order == [
        "produce0", "consume0",
        "produce1", "consume1",
        "produce2", "consume2",
        "produce3", "consume3",
    ]


def test_generator_memory_is_independent_of_length():
    small = (i for i in range(10))
    large = (i for i in range(10_000_000))
    assert sys.getsizeof(small) == sys.getsizeof(large)
    assert sys.getsizeof(large) < sys.getsizeof([0] * 10_000)
