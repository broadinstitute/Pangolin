import logging
import time
from typing import Dict, Tuple

import gffutils
from intervaltree import IntervalTree

logger = logging.getLogger(__name__)


class GeneAnnotator:
    def __init__(self, annotation_file: str, use_cache: bool = True):
        self.use_cache = use_cache
        self.gtf = None
        self.trees = None

        if use_cache:
            self.trees = self._load_data(annotation_file)
        else:
            self.gtf = gffutils.FeatureDB(annotation_file)

    def _load_data(self, annotation_file: str) -> Dict[str, IntervalTree]:
        load_time = time.time()
        gtf = gffutils.FeatureDB(annotation_file)
        trees = {}
        for gene in gtf.features_of_type("gene"):
            if gene.seqid not in trees:
                trees[gene.seqid] = IntervalTree()
            exons = []
            for exon in gtf.children(gene, featuretype="exon"):
                exons.extend([exon[3], exon[4]])
            trees[gene.seqid][gene.start : gene.stop] = (gene.id, gene.strand, exons)
        logger.debug(f"Load cached db: {time.time() - load_time:.5f}s")
        return trees

    def get_genes(self, chrom, pos) -> Tuple[Dict, Dict]:
        if self.use_cache:
            return self.get_cached_genes(chrom, pos)
        return self.get_db_genes(chrom, pos)

    def get_cached_genes(self, chrom: str, pos: int) -> Tuple[Dict, Dict]:
        genes = self.trees[chrom][pos - 1]

        genes_pos, genes_neg = {}, {}
        for gene in genes:
            gene_id, strand, exons = gene.data
            if strand == "+":
                genes_pos[gene_id] = exons
            elif strand == "-":
                genes_neg[gene_id] = exons

        return genes_pos, genes_neg

    def get_db_genes(self, chrom: str, pos: int) -> Tuple[Dict, Dict]:
        genes = self.gtf.region((chrom, pos - 1, pos - 1), featuretype="gene")
        genes_pos, genes_neg = {}, {}

        for gene in genes:
            if gene[3] > pos or gene[4] < pos:
                continue
            gene_id = gene["gene_id"][0]
            exons = []
            for exon in self.gtf.children(gene, featuretype="exon"):
                exons.extend([exon[3], exon[4]])
            if gene[6] == "+":
                genes_pos[gene_id] = exons
            elif gene[6] == "-":
                genes_neg[gene_id] = exons

        return (genes_pos, genes_neg)
