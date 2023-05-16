"""script for running pangolin from the command line"""
import argparse
import logging
from dataclasses import asdict

import torch

from pangolin.data_models import AppConfig

import time

from pangolin.processors import process_variants_file


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option)."
    )
    parser.add_argument(
        "reference_file", help="FASTA file containing a reference genome sequence."
    )
    parser.add_argument(
        "annotation_file",
        help="gffutils database file. Can be generated using create_db.py.",
    )
    parser.add_argument("output_file", help="Name of output file")
    parser.add_argument(
        "-c",
        "--column_ids",
        default="CHROM,POS,REF,ALT",
        help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
        "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)",
    )
    parser.add_argument(
        "-m",
        "--mask",
        default="True",
        choices=["False", "True"],
        help="If True, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will be set to 0. (Default: True)",
    )
    parser.add_argument(
        "-s",
        "--score_cutoff",
        type=float,
        help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=50,
        help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=0,
        help="Number of variants to batch together",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Enable additional debugging output",
    )
    parser.add_argument(
        "--enable_gtf_cache",
        default=False,
        action="store_true",
        help="Enable GTF db in memory caching, useful for large batches",
    )
    parser.add_argument(
        "--score_exons",
        default="False",
        choices=["False", "True"],
        help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...",
    )
    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(
        format="%(processName)s %(threadName)s %(asctime)s %(levelname)s %(name)s: - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )

    start_time = time.time()

    if torch.cuda.is_available():
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")

    app_config = AppConfig.from_args(args)

    logger.info(f"Using config : {asdict(app_config)}")

    process_variants_file(app_config)

    print(f"Execution time in seconds: {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()
