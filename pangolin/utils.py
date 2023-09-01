import logging
import re
import time
from typing import Tuple

import numpy as np
from pyfaidx import Fasta
import torch

from pangolin.data_models import (
    Variant,
    PreppedVariant,
    VariantEncodings,
    AppConfig,
    TimingDetails,
)
from pangolin.genes import GeneAnnotator

logger = logging.getLogger(__name__)


IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

SEQ_PATTERN = re.compile("^[ACTGN]+$")


def compute_score(ref_seq, alt_seq, strand, d, models):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

    if torch.cuda.is_available():
        ref_seq = ref_seq.to(torch.device("cuda"))
        alt_seq = alt_seq.to(torch.device("cuda"))

    pangolin = []
    for j in range(4):
        score = []
        for model in models[3 * j : 3 * j + 3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1, 4, 7, 10][j], :].cpu().numpy()
                alt = model(alt_seq)[0][[1, 4, 7, 10][j], :].cpu().numpy()
                if strand == "-":
                    ref = ref[::-1]
                    alt = alt[::-1]
                l = 2 * d + 1
                ndiff = np.abs(len(ref) - len(alt))
                if len(ref) > len(alt):
                    alt = np.concatenate(
                        [alt[0 : l // 2 + 1], np.zeros(ndiff), alt[l // 2 + 1 :]]
                    )
                elif len(ref) < len(alt):
                    alt = np.concatenate(
                        [
                            alt[0 : l // 2],
                            np.max(alt[l // 2 : l // 2 + ndiff + 1], keepdims=True),
                            alt[l // 2 + ndiff + 1 :],
                        ]
                    )
                score.append(alt - ref)
        pangolin.append(np.mean(score, axis=0))

    pangolin = np.array(pangolin)
    loss = pangolin[np.argmin(pangolin, axis=0), np.arange(pangolin.shape[1])]
    gain = pangolin[np.argmax(pangolin, axis=0), np.arange(pangolin.shape[1])]
    return loss, gain


def combine_scores(
    variant_pos,
    genes_pos,
    loss_pos,
    gain_pos,
    genes_neg,
    loss_neg,
    gain_neg,
    app_config: AppConfig,
) -> str:
    all_gene_scores = []

    for genes, orig_loss, orig_gain in (
        (genes_pos, loss_pos, gain_pos),
        (genes_neg, loss_neg, gain_neg),
    ):
        # The index values of `gain` and `loss` are relative base positions (relative to left side of distance window).
        # The value of `app_config.distance` is also a relative base position and is equal to the
        # "Number of bases on either side of the variant for which splice scores should be calculated."
        # When app_config.distance is d, gain and loss are arrays have size `2d + 1`.
        for gene, positions in genes.items():
            warnings = "Warnings:"
            positions = np.array(positions)
            # Convert to relative base positions (relative to left side of distance window)
            positions = positions - (variant_pos - app_config.distance)

            # Make copies of the loss/gain for each gene to avoid overwriting data between genes
            loss = np.copy(orig_loss)
            gain = np.copy(orig_gain)

            # Mask gain and loss scores for positions that occur in the provided array of positions
            if app_config.mask == "True" and len(positions) != 0:
                # find positions that are within the distance window
                positions_filt = positions[(positions >= 0) & (positions < len(loss))]
                # set splice gain at annotated sites to 0
                gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                # set splice loss at unannotated sites to 0
                not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                loss[not_positions] = np.maximum(loss[not_positions], 0)

            elif app_config.mask == "True":
                warnings += "NoAnnotatedSitesToMaskForThisGene"
                loss[:] = np.maximum(loss[:], 0)

            if app_config.score_exons == "True":
                scores1 = gene + "_sites1|"
                scores2 = gene + "_sites2|"

                for i in range(len(positions) // 2):
                    p1, p2 = positions[2 * i], positions[2 * i + 1]
                    if p1 < 0 or p1 >= len(loss):
                        s1 = "NA"
                    else:
                        s1 = [loss[p1], gain[p1]]
                        s1 = round(s1[np.argmax(np.abs(s1))], 2)
                    if p2 < 0 or p2 >= len(loss):
                        s2 = "NA"
                    else:
                        s2 = [loss[p2], gain[p2]]
                        s2 = round(s2[np.argmax(np.abs(s2))], 2)
                    if s1 == "NA" and s2 == "NA":
                        continue
                    scores1 += "%s:%s|" % (p1 - app_config.distance, s1)
                    scores2 += "%s:%s|" % (p2 - app_config.distance, s2)
                score = scores1 + scores2

            elif app_config.score_cutoff != None:
                score = gene + "|"
                l, g = (
                    np.where(loss <= -app_config.score_cutoff)[0],
                    np.where(gain >= app_config.score_cutoff)[0],
                )
                for p, s in zip(
                    np.concatenate([g - app_config.distance, l - app_config.distance]),
                    np.concatenate([gain[g], loss[l]]),
                ):
                    score += "%s:%s|" % (p, round(s, 2))

            else:
                score = gene + "|"
                l, g = (
                    np.argmin(loss),
                    np.argmax(gain),
                )
                # left side of each colon: relative base position (relative to variant position)
                # right side of each colon: score for that position
                score += "%s:%s|%s:%s|" % (
                    g - app_config.distance,
                    round(gain[g], 2),
                    l - app_config.distance,
                    round(loss[l], 2),
                )

            score += warnings
            all_gene_scores.append(score.strip("|"))

    return "||".join(all_gene_scores)


def one_hot_encode(seq, strand):
    seq = seq.upper().replace("A", "1").replace("C", "2")
    seq = seq.replace("G", "3").replace("T", "4").replace("N", "0")
    if strand == "+":
        seq = np.asarray(list(map(int, seq)))
    elif strand == "-":
        seq = np.asarray(list(map(int, seq[::-1])))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype("int8")]


def encode_seqs(ref_seq, alt_seq, strand):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()
    return ref_seq, alt_seq


def prepare_variant(
    variant: Variant, gene_annotator: GeneAnnotator, fasta: Fasta, distance: int
) -> Tuple[PreppedVariant, TimingDetails]:
    chr = variant.chr
    pos = variant.pos
    ref = variant.ref
    alt = variant.alt

    empty_timing = TimingDetails()

    skip_message = ""
    seq_time = time.time()
    if (
        len(set("ACGT").intersection(set(ref))) == 0
        or len(set("ACGT").intersection(set(alt))) == 0
        or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt))
    ):
        skip_message = "Variant format not supported."
    elif len(ref) > 2 * distance:
        skip_message = "Deletion too large"

    if skip_message:
        return (
            PreppedVariant.with_skip_message(
                variant=variant, skip_message=skip_message
            ),
            empty_timing,
        )

    # try to make vcf chromosomes compatible with reference chromosomes
    fasta_keys = fasta.keys()
    if chr not in fasta_keys and "chr" + chr in fasta_keys:
        variant.chr = "chr" + chr
    elif chr not in fasta_keys and chr[3:] in fasta_keys:
        variant.chr = chr[3:]

    seq = ""
    try:
        seq = fasta[chr][pos - 5001 - distance : pos + len(ref) + 4999 + distance].seq
    except Exception as e:
        logger.exception(e)
        skip_message = (
            "Could not get sequence, possibly because the variant is too close to chromosome ends. "
            "See error message above."
        )
        if skip_message:
            return (
                PreppedVariant.with_skip_message(
                    variant=variant, skip_message=skip_message
                ),
                empty_timing,
            )

    # This check ensures that only ACTGN characters are in the padded sequence from the FASTA file. If there
    # are any other characters, the downstream encoding will fail in one_hot_encode
    if re.search(SEQ_PATTERN, seq.upper()) is None:
        skip_message = f"Unsupported sequences in ref seq from fasta, found bases: {set(seq).difference(set('ACTGN'))}"
        return (
            PreppedVariant.with_skip_message(
                variant=variant, skip_message=skip_message
            ),
            empty_timing,
        )

    if seq[5000 + distance : 5000 + distance + len(ref)].upper() != ref:
        ref_base = seq[5000 + distance : 5000 + distance + len(ref)]
        skip_message = f"Mismatch between FASTA (ref base: {ref_base}) and variant file (ref base: {ref})."
        return (
            PreppedVariant.with_skip_message(
                variant=variant, skip_message=skip_message
            ),
            empty_timing,
        )

    ref_seq = seq
    alt_seq = seq[: 5000 + distance] + alt + seq[5000 + distance + len(ref) :]
    total_seq_time = time.time() - seq_time

    gene_time = time.time()
    genes_pos, genes_neg = gene_annotator.get_genes(chr, pos)
    if len(genes_pos) + len(genes_neg) == 0:
        skip_message = (
            "Variant not contained in a gene body. Do GTF/FASTA chromosome names match?"
        )
        return (
            PreppedVariant.with_skip_message(
                variant=variant, skip_message=skip_message
            ),
            empty_timing,
        )
    total_gene_time = time.time() - gene_time

    encode_time = time.time()
    encoded_ref_pos, encoded_alt_pos, encoded_ref_neg, encoded_alt_neg = "", "", "", ""
    if len(genes_pos) > 0:
        encoded_ref_pos, encoded_alt_pos = encode_seqs(ref_seq, alt_seq, "+")
    if len(genes_neg) > 0:
        encoded_ref_neg, encoded_alt_neg = encode_seqs(ref_seq, alt_seq, "-")
    total_encode_time = time.time() - encode_time

    prep_timing = TimingDetails(
        seq_time=total_seq_time,
        gene_time=total_gene_time,
        encode_time=total_encode_time,
    )

    return (
        PreppedVariant(
            variant=variant,
            genes_pos=genes_pos,
            genes_neg=genes_neg,
            encodings=VariantEncodings(
                encoded_ref_neg=encoded_ref_neg,
                encoded_ref_pos=encoded_ref_pos,
                encoded_alt_pos=encoded_alt_pos,
                encoded_alt_neg=encoded_alt_neg,
            ),
        ),
        prep_timing,
    )
