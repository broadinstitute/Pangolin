# Original source code modified to add prediction batching support and bugfixes by Invitae in 2023.
# Modifications copyright (c) 2023 Invitae Corporation.

import logging
import time

import numpy as np
import pyfastx
import torch
from typing import List, Dict, Tuple

from pangolin.data_models import (
    Variant,
    PreppedVariant,
    BatchLookupIndex,
    AppConfig,
    SequenceType,
)
from pangolin.genes import GeneAnnotator
from pangolin.utils import combine_scores, prepare_variant

logger = logging.getLogger(__name__)


class PredictionBatch:
    def __init__(self, models: List, app_config: AppConfig):
        self.app_config = app_config
        self.models = models
        self.gene_annotator = GeneAnnotator(
            self.app_config.annotation_file,
            use_cache=self.app_config.enable_gtf_cache,
        )
        self.fasta = pyfastx.Fasta(self.app_config.reference_file)

        self.batches = {}
        self.variants: List[Variant] = []
        self.prepared_records: List[PreppedVariant] = []

        self.batch_count = 0
        self.total_records = 0

        self.prep_total_time = None
        self.batch_start_time = None

        # Flag to know when the batch was run
        self.did_run_predictions = False

        logger.debug(f"Batch init with batch size: {self.app_config.batch_size}")

    def batch_variant(self, prepped_variant: PreppedVariant) -> List[BatchLookupIndex]:
        # Skip batching this variant if it wasn't encoded for validation reasons
        if not prepped_variant.encodings:
            return []

        encoded_ref_pos = (
            prepped_variant.encodings.encoded_ref_pos
            if prepped_variant.encodings
            else ""
        )
        encoded_alt_pos = (
            prepped_variant.encodings.encoded_alt_pos
            if prepped_variant.encodings
            else ""
        )
        encoded_ref_neg = (
            prepped_variant.encodings.encoded_ref_neg
            if prepped_variant.encodings
            else ""
        )
        encoded_alt_neg = (
            prepped_variant.encodings.encoded_alt_neg
            if prepped_variant.encodings
            else ""
        )

        batch_lookup_indexes = []

        for var_type, encoded_seq in zip(
            (
                SequenceType.POS_REF,
                SequenceType.POS_ALT,
                SequenceType.NEG_REF,
                SequenceType.NEG_ALT,
            ),
            (encoded_ref_pos, encoded_alt_pos, encoded_ref_neg, encoded_alt_neg),
        ):
            if len(encoded_seq) == 0:
                # Add BatchLookupIndex with zeros so when the batch collects the outputs
                # it knows that there is no prediction for this record
                batch_lookup_indexes.append(BatchLookupIndex(var_type, 0, 0))
                continue

            # Iterate over the encoded sequence and drop into the correct batch by size and
            # create an index to use to pull out the result after batch is processed
            # for row in encoded_seq:
            # Extract the size of the sequence that was encoded to build a batch from
            tensor_size = encoded_seq.shape[2]

            # Create batch for this size
            if tensor_size not in self.batches:
                self.batches[tensor_size] = []

            # Add encoded record to batch
            self.batches[tensor_size].append(encoded_seq)

            # Get the index of the record we just added in the batch
            cur_batch_record_ix = len(self.batches[tensor_size]) - 1

            # Store a reference so we can pull out the prediction for this item from the batches
            batch_lookup_indexes.append(
                BatchLookupIndex(var_type, tensor_size, cur_batch_record_ix)
            )

        return batch_lookup_indexes

    def prep_all_variants(self) -> None:
        prep_time = time.time()
        total_seq_time = 0
        total_encode_time = 0
        total_gene_time = 0
        for variant in self.variants:
            prepared_record, timing = prepare_variant(
                variant,
                self.gene_annotator,
                self.fasta,
                self.app_config.distance,
            )
            if prepared_record.skip_message:
                logger.debug(prepared_record.skip_message)
            total_seq_time += timing.seq_time
            total_encode_time += timing.encode_time
            total_gene_time += timing.gene_time
            self.prepared_records.append(prepared_record)
        self.prep_total_time = time.time() - prep_time
        logger.debug(f"Total seq time: {total_seq_time:.5f}s")
        logger.debug(f"Total gene time: {total_gene_time:.5f}s")
        logger.debug(f"Total encode time: {total_encode_time:.5f}s")
        logger.debug(f"Prep variant time: {self.prep_total_time:.5f}s")

        # Put the variants into buckets
        for prepped_variant in self.prepared_records:
            prepped_variant.locations = self.batch_variant(prepped_variant)

    def add_variant(self, variant: Variant) -> None:
        self.total_records += 1
        self.variants.append(variant)
        self.did_run_predictions = False

        # Once we fill the batch, process the records
        if len(self.variants) >= self.app_config.batch_size:
            logger.debug(f"Finished collected variants in batch: {len(self.variants)}")
            self.run_batch()
            self.did_run_predictions = True

    def run_batch(self) -> None:
        self.batch_start_time = time.time()
        self.prep_all_variants()
        self._process_batch()

    def finish(self) -> None:
        logger.debug("Finish")

        if len(self.variants) == 0:
            logger.debug("No variants left to process")
            return

        # Run remaining variants
        self.run_batch()

    def run_predictions(self, batch) -> List:
        batch_preds = []
        if torch.cuda.is_available():
            batch = batch.to(torch.device("cuda"))
        for j in range(4):
            for i, model in enumerate(self.models[3 * j : 3 * j + 3]):
                with torch.no_grad():
                    preds = model(batch)
                    batch_preds.append(preds)
        return batch_preds

    def _process_batch(self) -> None:
        start = time.time()
        total_batch_predictions = 0
        self.batch_count += 1
        logger.debug(f"Starting process_batch ({self.batch_count})")

        batch_sizes = [
            "{}:{}".format(tensor_size, len(batch))
            for tensor_size, batch in self.batches.items()
        ]
        logger.debug("Batch Sizes: {}".format(batch_sizes))

        batch_preds = {}
        for tensor_size, batch in self.batches.items():
            # Convert list of encodings into a proper sized numpy matrix
            prediction_batch = np.concatenate(batch, axis=0)
            torched = torch.from_numpy(prediction_batch).float()
            batch_preds[tensor_size] = self.run_predictions(torched)

        for prepped_record in self.prepared_records:
            (
                prepped_record.loss_pos,
                prepped_record.gain_pos,
            ) = self._get_score_from_batch(prepped_record, batch_preds, "+")
            (
                prepped_record.loss_neg,
                prepped_record.gain_neg,
            ) = self._get_score_from_batch(prepped_record, batch_preds, "-")
            prepped_record.score = self.calculate_score(prepped_record)
            total_batch_predictions += 1

        duration = time.time() - start
        logger.debug(f"Batch time: {duration:0.2f}s")
        batch_duration = time.time() - self.batch_start_time
        preds_per_sec = total_batch_predictions / batch_duration
        preds_per_hour = preds_per_sec * 60 * 60
        logger.info(
            f"Finished batch {self.batch_count}: Total Time {batch_duration:0.2f}s, Prep Time: {self.prep_total_time:0.2f}s, Preds/Hour: {preds_per_hour:0.0f}, Records: {self.total_records}"
        )

    def _get_score_from_batch(
        self, prepped_record: PreppedVariant, batch_preds: Dict[int, List], strand: str
    ) -> Tuple:
        if len(prepped_record.locations) == 0:
            return None, None

        # Get the lookup locations of the ref and alt values
        ref_location = (
            prepped_record.locations[SequenceType.POS_REF.value]
            if strand == "+"
            else prepped_record.locations[SequenceType.NEG_REF.value]
        )
        alt_location = (
            prepped_record.locations[SequenceType.POS_ALT.value]
            if strand == "+"
            else prepped_record.locations[SequenceType.NEG_ALT.value]
        )

        if ref_location.tensor_size == 0 and alt_location.tensor_size == 0:
            return None, None

        ix = 0
        pangolin = []
        for j in range(4):
            scores = []
            for _ in self.models[3 * j : 3 * j + 3]:
                # Pull out predictions from the batch
                ref_prediction = batch_preds[ref_location.tensor_size][ix]
                alt_prediction = batch_preds[alt_location.tensor_size][ix]

                # Bring data back to CPU
                ref = (
                    ref_prediction[ref_location.batch_index][[1, 4, 7, 10][j], :]
                    .cpu()
                    .numpy()
                )
                alt = (
                    alt_prediction[alt_location.batch_index][[1, 4, 7, 10][j], :]
                    .cpu()
                    .numpy()
                )
                if strand == "-":
                    ref = ref[::-1]
                    alt = alt[::-1]
                l = 2 * self.app_config.distance + 1
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
                score = alt - ref
                scores.append(score)
                ix += 1
            pangolin.append(np.mean(scores, axis=0))

        pangolin = np.array(pangolin)
        loss = pangolin[np.argmin(pangolin, axis=0), np.arange(pangolin.shape[1])]
        gain = pangolin[np.argmax(pangolin, axis=0), np.arange(pangolin.shape[1])]
        return loss, gain

    def calculate_score(self, variant: PreppedVariant) -> str:
        if len(variant.locations) == 0:
            return ""
        scores = combine_scores(
            variant.variant.pos,
            variant.genes_pos,
            variant.loss_pos,
            variant.gain_pos,
            variant.genes_neg,
            variant.loss_neg,
            variant.gain_neg,
            self.app_config,
        )
        return scores

    def clear_batch(self) -> None:
        self.batches.clear()
        del self.variants[:]
        del self.prepared_records[:]
