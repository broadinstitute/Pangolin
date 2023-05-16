"""Script to define data models used in the pangolin package."""
import dataclasses
from enum import Enum
from typing import Optional, List

from torch._C._te import Tensor


class SequenceType(Enum):
    """Enum to represent the type of sequence being encoded."""

    POS_REF = 0
    POS_ALT = 1
    NEG_REF = 2
    NEG_ALT = 3


@dataclasses.dataclass
class TimingDetails:
    """Dataclass to hold timing details for a variant."""

    seq_time: float = 0
    encode_time: float = 0
    gene_time: float = 0


@dataclasses.dataclass
class BatchLookupIndex:
    """Dataclass to hold the lookup index for a variant in the batching."""

    sequence_type: SequenceType
    tensor_size: int
    batch_index: int


@dataclasses.dataclass
class AppConfig:
    """Dataclass to hold the configuration for the app."""

    variant_file: str
    output_file: str
    reference_file: str
    annotation_file: str
    batch_size: int
    distance: int
    mask: str
    score_exons: str
    column_ids: str
    score_cutoff: Optional[float]
    enable_gtf_cache: bool

    @classmethod
    def from_args(cls, args) -> "AppConfig":
        """Create an AppConfig from the command line arguments."""
        return cls(
            variant_file=args.variant_file,
            output_file=args.output_file,
            reference_file=args.reference_file,
            annotation_file=args.annotation_file,
            batch_size=args.batch_size,
            distance=args.distance,
            score_cutoff=args.score_cutoff,
            mask=args.mask,
            score_exons=args.score_exons,
            column_ids=args.column_ids,
            enable_gtf_cache=args.enable_gtf_cache,
        )


@dataclasses.dataclass
class Variant:
    """Dataclass to hold the details of a variant."""

    lnum: int
    chr: str
    pos: int
    ref: str
    alt: str
    id: Optional[int] = None


@dataclasses.dataclass
class VariantEncodings:
    """Dataclass to hold the encodings for a variant."""

    encoded_ref_pos: Tensor
    encoded_alt_pos: Tensor
    encoded_ref_neg: Tensor
    encoded_alt_neg: Tensor


@dataclasses.dataclass
class PreppedVariant:
    """Dataclass to hold the details of a variant after it has been prepped."""

    variant: Variant
    score: str = ""
    skip_message: str = ""
    locations: Optional[List[BatchLookupIndex]] = None
    encodings: Optional[VariantEncodings] = None
    genes_pos: Optional[List] = None
    genes_neg: Optional[List] = None
    loss_pos: Optional[List] = None
    gain_pos: Optional[List] = None
    loss_neg: Optional[List] = None
    gain_neg: Optional[List] = None

    @classmethod
    def with_skip_message(cls, variant: Variant, skip_message: str) -> "PreppedVariant":
        """Create a PreppedVariant with a skip message."""
        return cls(variant=variant, skip_message=skip_message, locations=[])
