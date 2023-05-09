import difflib
import sys
import tempfile

import pytest

from pangolin.data_models import AppConfig
from pangolin.processors import process_variants_file


def build_config(input_file: str, output_file: str, batch_size: int = 1) -> AppConfig:
    # This is just a download of the chr19 reference genome
    ref_file = "tests/data/reference/chr19.fa.gz"

    # The chr19_genes.gtf.gz is just a
    # This gtf file was built with the following set of commands. I hardcoded in the explicit gene names used
    # in the test files to reduce the size of the database and files
    # gzcat tests/data/reference/chr19_genes.gtf.gz | grep 'PNKP\|ELANE\|STK11\|CBARP' > tests/data/reference/chr19_genes_filtered.gtf
    # python scripts/create_db.py tests/data/reference/chr19_genes_filtered.gtf --filter None
    gtf_file = "tests/data/reference/chr19_genes_filtered.db"

    app_config = AppConfig(
        variant_file=input_file,
        output_file=output_file,
        reference_file=ref_file,
        annotation_file=gtf_file,
        batch_size=batch_size,
        distance=200,
        score_cutoff=None,
        mask="True",
        score_exons="False",
        column_ids="CHROM,POS,REF,ALT",
        enable_gtf_cache=True,
    )
    return app_config


def run_pangolin(input_file, expected_file, batch_size: int = 0, suffix: str = ""):
    with tempfile.NamedTemporaryFile(suffix=suffix) as fh:
        output_file = fh.name
        config = build_config(input_file, output_file, batch_size)
        process_variants_file(config)
        with open(output_file) as out_fh:
            batch_file_contents = out_fh.readlines()

    with open(expected_file) as fh:
        expected_file_contents = fh.readlines()

    if expected_file_contents != batch_file_contents:
        sys.stdout.writelines(
            difflib.unified_diff(expected_file_contents, batch_file_contents)
        )
    assert expected_file_contents == batch_file_contents


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        1,
    ],
)
def test_batch_vcf(batch_size):
    input_file = "tests/data/small.vcf"
    expected_file = "tests/data/expected/small_out.vcf"
    run_pangolin(input_file, expected_file, batch_size=batch_size, suffix=".vcf")


@pytest.mark.parametrize(
    "batch_size",
    [0, 2, 3, 5],
)
def test_batch_csv(batch_size):
    input_file = "tests/data/small.csv"
    expected_file = "tests/data/expected/small_out.csv"
    run_pangolin(input_file, expected_file, batch_size=batch_size, suffix=".csv")
