# Original source code modified to add prediction batching support and bugfixes by Invitae in 2023.
# Modifications copyright (c) 2023 Invitae Corporation.

import logging
from typing import Callable, List, Union

import pandas as pd
import pysam
import typing

from pysam import VariantFile

from pangolin.batch import PredictionBatch
from pangolin.model import load_models
from pangolin.data_models import Variant, AppConfig

logger = logging.getLogger(__name__)


def process_variants_file(app_config: AppConfig) -> None:
    models = load_models()
    batch = PredictionBatch(models, app_config)
    if app_config.variant_file.endswith(".vcf"):
        process_vcf(batch, app_config)
    elif app_config.variant_file.endswith(".csv"):
        process_csv(batch, app_config)
    else:
        raise RuntimeError("ERROR, variant_file needs to be a CSV or VCF.")


def handle_batch(
    batch: PredictionBatch,
    original_records: List,
    writer: Callable,
    fout: Union[typing.TextIO, VariantFile],
) -> None:
    for prepared_record, original_record in zip(
        batch.prepared_records, original_records
    ):
        writer(original_record, prepared_record.score, fout)


def vcf_writer(original_record, score: str, fout: VariantFile) -> None:
    if score != "":
        original_record.info["Pangolin"] = score
    fout.write(original_record)


def csv_writer(original_record, score: str, fout: typing.TextIO) -> None:
    if score == "":
        fout.write(
            ",".join(original_record.to_csv(header=False, index=False).split("\n"))
            + "\n"
        )
    else:
        fout.write(
            ",".join(original_record.to_csv(header=False, index=False).split("\n"))
            + score
            + "\n"
        )


def process_vcf(batch: PredictionBatch, app_config: AppConfig):
    input_vcf = pysam.VariantFile(app_config.variant_file)
    header = input_vcf.header
    header.add_line(
        '##INFO=<ID=Pangolin,Number=.,Type=String,Description="Pangolin splice scores. '
        'Format: gene|pos:score_change|pos:score_change|...">'
    )
    fout = pysam.VariantFile(app_config.output_file, "w", header=header)

    original_records = []
    for i, variant in enumerate(input_vcf):
        # Store original VCF row
        original_records.append(variant)
        # NOTE: Only single alts are supported here
        if len(variant.alts) > 1:
            raise RuntimeError(
                f"Only single ALTs are supported for VCF predictions"
            )
        v = Variant(
            i,
            chr=str(variant.chrom),
            pos=int(variant.pos),
            ref=variant.ref,
            alt=variant.alts[0],
        )
        batch.add_variant(v)
        if batch.did_run_predictions:
            handle_batch(batch, original_records, vcf_writer, fout)
            original_records.clear()
            batch.clear_batch()

    if app_config.batch_size > 0:
        batch.finish()
        handle_batch(batch, original_records, vcf_writer, fout)

    fout.close()
    print(f"Wrote results to: {app_config.output_file}")


def process_csv(batch: PredictionBatch, app_config: AppConfig):
    col_ids = app_config.column_ids.split(",")
    variants = pd.read_csv(app_config.variant_file, header=0)
    fout = open(app_config.output_file, "w")
    fout.write(",".join(variants.columns) + ",Pangolin\n")
    fout.flush()

    # Store original record here to use again when batching is completed
    original_records = []

    for lnum, variant in variants.iterrows():
        lnum = typing.cast(int, lnum)  # Used to solve type hinting issues
        chr, pos, ref, alt = variant[col_ids]
        ref, alt = ref.upper(), alt.upper()

        # Store original CSV record
        original_records.append(variant)
        v = Variant(lnum=lnum, chr=str(chr), pos=int(pos), ref=ref, alt=alt)

        batch.add_variant(v)
        if batch.did_run_predictions:
            handle_batch(batch, original_records, csv_writer, fout)
            original_records.clear()
            batch.clear_batch()

    if app_config.batch_size > 0:
        batch.finish()
        handle_batch(batch, original_records, csv_writer, fout)

    fout.close()
    print(f"Wrote results to: {app_config.output_file}")
