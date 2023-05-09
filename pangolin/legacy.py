import pyfastx

from pangolin.data_models import AppConfig
from pangolin.utils import compute_score, combine_scores


def get_genes(chrom, pos, gtf):
    genes = gtf.region((chrom, pos - 1, pos - 1), featuretype="gene")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        gene_id = gene["gene_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == "+":
            genes_pos[gene_id] = exons
        elif gene[6] == "-":
            genes_neg[gene_id] = exons

    return (genes_pos, genes_neg)


def process_variant_legacy(
    lnum, chr, pos, ref, alt, gtf, models, app_config: AppConfig
):
    d = app_config.distance

    if (
        len(set("ACGT").intersection(set(ref))) == 0
        or len(set("ACGT").intersection(set(alt))) == 0
        or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt))
    ):
        print(
            "[Line %s]" % lnum,
            "WARNING, skipping variant: Variant format not supported.",
        )
        return -1
    elif len(ref) > 2 * d:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Deletion too large")
        return -1

    fasta = pyfastx.Fasta(app_config.reference_file)
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr" + chr in fasta.keys():
        chr = "chr" + chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos - 5001 - d : pos + len(ref) + 4999 + d].seq
    except Exception as e:
        print(e)
        print(
            "[Line %s]" % lnum,
            "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
            "See error message above.",
        )
        return -1

    if seq[5000 + d : 5000 + d + len(ref)].upper() != ref:
        print(
            "[Line %s]" % lnum,
            "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
            % (seq[5000 + d : 5000 + d + len(ref)], ref),
        )
        return -1

    ref_seq = seq
    alt_seq = seq[: 5000 + d] + alt + seq[5000 + d + len(ref) :]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos) + len(genes_neg) == 0:
        print(
            "[Line %s]" % lnum,
            "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?",
        )
        return -1

    # get splice scores
    loss_pos, gain_pos = None, None
    if len(genes_pos) > 0:
        loss_pos, gain_pos = compute_score(ref_seq, alt_seq, "+", d, models)
    loss_neg, gain_neg = None, None
    if len(genes_neg) > 0:
        loss_neg, gain_neg = compute_score(ref_seq, alt_seq, "-", d, models)

    scores = combine_scores(
        pos, genes_pos, loss_pos, gain_pos, genes_neg, loss_neg, gain_neg, app_config
    )
    return scores
