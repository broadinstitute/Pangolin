from collections import namedtuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from pangolin.data_models import Variant
from pangolin.utils import prepare_variant, combine_scores


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


@pytest.mark.parametrize("genes, gain, loss, expected_score", [
    pytest.param(
        {"GENE1": []},
        [0.0, 0.0, 0.0, 0.7, 0.0],  # gain array not masked
        [0.0, -0.5, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|1:0.7|-2:0.0|Warnings:NoAnnotatedSitesToMaskForThisGene",
        id="no positions, gain array not masked, warning placed"
    ),
    pytest.param(
        {"GENE1": [5, 15]},
        [0.0, 0.0, 0.0, 0.0, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, 0.0, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-2:0.0|-2:0.0|Warnings:SingleExonTranscript",
        id="no position inside window, zero gain and loss arrays"
    ),
    pytest.param(
        {"GENE1": [5, 15]},
        [0.0, 0.0, 0.0, 0.7, -0.7],  # gain array masked to [0.0, 0.0, 0.0, 0.0, -0.7]
        [0.5, -0.5, 0.0, 0.0, 0.0],  # loss array masked to [0.5, 0.0, 0.0, 0.0, 0.0]
        "GENE1|1:0.7|-1:0.0|Warnings:SingleExonTranscript",
        id="no position inside window, gain and loss arrays"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.0, 0.0, 0.0, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, 0.0, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-2:0.0|-2:0.0|Warnings:",
        id="zero gain and loss arrays"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.0, 0.0, 0.7, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, 0.0, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-2:0.0|-2:0.0|Warnings:",
        id="gain score is masked"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.0, 0.0, 0.0, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, -0.5, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-2:0.0|-2:0.0|Warnings:",
        id="loss score is masked"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.0, 0.0, 0.7, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, -0.5, 0.0, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-2:0.0|-2:0.0|Warnings:",
        id="gain and loss scores are masked"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.0, 0.0, 0.7, 0.0],  # gain array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        [0.0, 0.0, 0.0, -0.5, 0.0],  # loss array masked to [0.0, 0.0, 0.0, -0.5, 0.0]
        "GENE1|-2:0.0|1:-0.5|Warnings:",
        id="gain score is masked, loss score is protected"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.7, 0.0, 0.0, 0.0],  # gain array masked to [0.0, 0.7, 0.0, 0.0, 0.0]
        [0.0, 0.0, -0.5, 0.0, 0.0],  # loss array masked to [0.0, 0.0, 0.0, 0.0, 0.0]
        "GENE1|-1:0.7|-2:0.0|Warnings:",
        id="gain score is not masked, loss score is masked"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.0, 0.7, 0.0, 0.0, 0.0],  # gain array masked to [0.0, 0.7, 0.0, 0.0, 0.0]
        [0.0, 0.0, 0.0, -0.5, 0.0],  # loss array masked to [0.0, 0.0, 0.0, -0.5, 0.0]
        "GENE1|-1:0.7|1:-0.5|Warnings:",
        id="gain score and loss score are not masked (loss score is protected)"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16]},
        [0.6, 0.7, 0.5, 0.4, 0.3],  # gain array masked to [0.6, 0.7, 0.5, 0.0, 0.3]
        [-0.3, -0.4, -0.4, -0.5, -0.3],  # loss array masked to [0.0, 0.0, 0.0, -0.5, 0.0]
        "GENE1|-1:0.7|1:-0.5|Warnings:",
        id="multiple gain and loss scores are not masked (loss score is protected)"
    ),
    # with multiple genes, gain and loss arrays are the same for all genes, but filter positions may differ by gene
    pytest.param(
        {"GENE1": [5, 11, 15, 16], "GENE2": [5, 11, 15, 16]},
        # for both genes, gain array masked to [0.6, 0.7, 0.5, 0.0, 0.3]
        [0.6, 0.7, 0.5, 0.4, 0.3],
        # for both genes, loss array masked to [0.0, 0.0, 0.0, -0.5, 0.0]
        [-0.3, -0.4, -0.4, -0.5, -0.3],
        "GENE1|-1:0.7|1:-0.5|Warnings:||GENE2|-1:0.7|1:-0.5|Warnings:",
        id="multiple genes, same positions"
    ),
    pytest.param(
        {"GENE1": [5, 11, 15, 16], "GENE2": [5, 9, 15, 16]},
        # for GENE1, gain array masked to [0.6, 0.7, 0.5, 0.0, 0.3]
        # for GENE2, gain array masked to [0.6, 0.0, 0.5, 0.4, 0.3]
        [0.6, 0.7, 0.5, 0.4, 0.3],
        # for GENE1, loss array masked to [0.0, 0.0, 0.0, -0.5, 0.0]
        # for GENE2, loss array masked to [0.0, -0.4, 0.0, 0.0, 0.0]
        [-0.3, -0.4, -0.4, -0.5, -0.3],
        "GENE1|-1:0.7|1:-0.5|Warnings:||GENE2|-2:0.6|-1:-0.4|Warnings:",
        id="multiple genes, different positions"
    ),
])
def test_combine_scores(genes, gain, loss, expected_score):
    """
    In all cases, distance is 2 and variant position is 10, so distance window is (8, 12) for gain and loss arrays.
    In gain array, positive numbers at positions inside the distance window will be masked to 0.
    In loss array, negative numbers at the remaining positions (all positions excluding the position masked in gain) are masked to 0.
    Gain score is the highest remaining score. Loss score is the lowest remaining score.
    Expected score is the gain score followed by the loss score.
    Position array for each gene represents exon boundaries, and therefore always has an even number length.
    Positive-strand and negative-strand cases behave the same way for the same genes/gain/loss data.
    """
    app_config = Mock(
        distance=2,
        mask="True",
        score_cutoff=None,
        score_exons="False",
    )
    variant_position = 10
    assert combine_scores(
        variant_pos=variant_position,
        genes_pos=genes,
        loss_pos=np.asarray(loss),
        gain_pos=np.asarray(gain),
        genes_neg={},
        loss_neg=np.asarray([]),
        gain_neg=np.asarray([]),
        app_config=app_config,
    ) == expected_score, "positive strand score is incorrect"
    assert combine_scores(
        variant_pos=variant_position,
        genes_pos={},
        loss_pos=np.asarray([]),
        gain_pos=np.asarray([]),
        genes_neg=genes,
        loss_neg=np.asarray(loss),
        gain_neg=np.asarray(gain),
        app_config=app_config,
    ) == expected_score, "negative strand score is incorrect"