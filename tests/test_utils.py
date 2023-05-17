from collections import namedtuple
from unittest.mock import MagicMock

import pytest

from pangolin.data_models import Variant
from pangolin.utils import prepare_variant


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("ACTGV", True),
        ("ACTG", False),
        ("AAAA", False),
        ("AAUAA", True),
        ("U", True),
    ],
)
def test_unsupported_ref_seq(seq, expected):
    variant = Variant(lnum=1, chr="chr1", pos=100, ref="C", alt="T")
    # Mock out the fasta seq so it returns the passed in sequence
    fasta = MagicMock()
    fasta.keys = MagicMock(return_value=[variant.chr])
    seq_obj = namedtuple("RefSeq", "seq")
    fasta[variant.chr].__getitem__ = MagicMock(return_value=seq_obj(seq))

    prepped_variant, _ = prepare_variant(variant, None, fasta, 500)
    skip_message_contains_text = (
        "Unsupported sequences in ref seq from fasta" in prepped_variant.skip_message
    )
    assert skip_message_contains_text == expected
