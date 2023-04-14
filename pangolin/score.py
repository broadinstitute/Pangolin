"""Score gene spices"""

import os
from pkg_resources import resource_filename
import numpy as np
import pandas as pd

import torch
import gffutils
import vcf
import pyfastx

#from pangolin.pangolin import Pangolin
from pangolin import model as pangolin_model


# TODO: is there another way to use an identity matrix faster?
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def one_hot_encode(seq, strand):
    # TODO: is there a sklearn method that's faster?
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


class PangolinScore:
    def __init__(self):
        self._variant_file    = None
        self._reference_file  = None
        self._annotation_file = None
        self._output_file     = None
        self._column_ids      = "CHROM,POS,REF,ALT"
        self._mask            = True
        self._score_cutoff    = None  # TODO: what is the correct default?
        self._distance        = 50
        self._score_exons     = False
        
        self.fp_model_weight_dir = None
        
        self.verbose = True
        
        self.gtf = None
        self.run_kind = 'CPU'
        if torch.cuda.is_available():
            self.run_kind = 'GPU'
        print(f"Using {self.run_kind} to compute.")
        
    @property
    def variant_file(self):
        return self._variant_file
    
    @variant_file.setter
    def variant_file(self, v):
        """"VCF or CSV file with a header (see COLUMN_IDS option).
        
        Parameters
        ----------
        v : str, filepath to VCF or CSV file
        """
        self._variant_file = v
        
    @property
    def reference_file(self):
        return self._reference_file
    
    @reference_file.setter
    def reference_file(self, rf):
        """FASTA file containing a reference genome sequence.
        
        Parameters
        ----------
        rf : str, filepath to FASTA file
        """
        self._reference_file = rf
        
    @property
    def annotation_file(self, af):
        return self._annotation_file
    
    @annotation_file.setter
    def annotation_file(self, af):
        """gffutils database file. Can be generated using create_db.py
        
        Parameters
        ----------
        af : str, filepath to gffutils database file
        """
        self._annotation_file = af
        
        if not os.path.exists(self._annotation_file):
            from pangolin import utils
            utils.create_db(self._annotation_file)
        
        try:
            self.gtf = gffutils.FeatureDB(self._annotation_file)
        except:
            raise Exception("ERROR, annotation_file could not be opened")
        
    @property
    def output_file(self, of):
        return self._output_file
    
    @output_file.setter
    def output_file(self, of):
        """Prefix for output file. Will be a VCF/CSV if 
        variant_file is VCF/CSV.
        
        Parameters
        ----------
        of : str, prefix
        """
        self._output_file = of
    
    @property
    def column_ids(self):
        return self._column_ids
    
    @column_ids.setter
    def column_ids(self, cids="CHROM,POS,REF,ALT"):
        """"If variant_file is a CSV, column IDs for: 
        column, default
        chromosome = CHROM, 
        variant position = POS, 
        reference bases = REF, 
        alternative bases = ALT.
        
        Parameters
        ----------
        cids : str, comma delimited string. Default: CHROM,POS,REF,ALT
        """
        self.column_ids = cids
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, m=True):
        """Splice gains (increases in score) at annotated splice sites 
        and 
        splice losses (decreases in score) 
        at unannotated splice sites will be set to 0.
        
        Parameters
        ----------
        m : bool, mask splice gains and losses, default = True
        """
        self._mask = m
        
    @property
    def score_cutoff(self):
        return self._score_cutoff
    
    @score_cutoff.setter
    def score_cutoff(self, sc):
        """Output all sites with absolute predicted change 
        in score >= cutoff, instead of only the maximum loss/gain sites.
        
        Parameters
        ----------
        sc : float, default: 50.0
        """
        self._score_cutoff = sc
    
    @property
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self, d):
        """Number of bases on either side of the variant 
        for which splice scores should be calculated.
        
        Parameters
        ----------
        d : int, default: 50
        """
        self._distance = d
    
    @property
    def score_exons(self):
        return self._score_exons
    
    @score_exons.setter
    def score_exons(self, se=False):
        """Output changes in score for both splice sites of annotated exons, 
        as long as one splice site is within the considered range (specified by -d). 
        Output will be: gene|site1_pos:score|site2_pos:score|...
        
        Parameters
        ----------
        se : bool, default: False
        """
        self._score_exons = se
        
    def load_weights(self):
        """Load pre-fit model weights.
        Assumes a directory contains files with format f'final.{j}.{i}.3.v2'
        where for i in [0,2,4,6] and j in range(1, 4)
        
        TODO: break out weights folder check: all combos or fail
            then just iterate through the list here
            
        TODO: this should also ideally only be done once per session,
            so maybe todo above is irrelevant
            
        TODO: if it really matters, use os.path.sep
        """
        model_list = []  # convert to null filled array of len(4*4) (ixj)
        for i in [0,2,4,6]:
            for j in range(1, 4):
                model_instance = pangolin_model.Pangolin(pangolin_model.L, 
                                                         pangolin_model.W, 
                                                         pangolin_model.AR)
                
                fp_weights = f"{self.fp_model_weight_dir}/final.{j}.{i}.3.v2"
                weights = torch.load(fp_weights)
                
                if self.run_kind == 'GPU':
                    model_instance.cuda()
                    weights = torch.load(fp_weights)
                else:
                    weights = torch.load(fp_weights, map_location=torch.device('cpu'))
                model_instance.load_state_dict(weights)
                model_instance.eval()
                model_list.append(model_instance)  #TODO: convert to null filled array
        self.model_list = model_list
    
    @staticmethod
    def get_genes(chr, pos, gtf):
        genes = gtf.region((chr, pos-1, pos-1), featuretype="gene")
        genes_pos, genes_neg = {}, {}

        for gene in genes:
            if gene[3] > pos or gene[4] < pos:
                continue
            gene_id = gene["gene_id"][0]
            exons = []
            for exon in gtf.children(gene, featuretype="exon"):
                exons.extend([exon[3], exon[4]])
            if gene[6] == '+':
                genes_pos[gene_id] = exons
            elif gene[6] == '-':
                genes_neg[gene_id] = exons

        return (genes_pos, genes_neg)
    
    def compute_score(self, ref_seq, alt_seq, strand, d, model_list):
        """Compute the loss and gain"""
        ref_seq = one_hot_encode(ref_seq, strand).T
        ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
        alt_seq = one_hot_encode(alt_seq, strand).T
        alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

        if torch.cuda.is_available():
            ref_seq = ref_seq.to(torch.device("cuda"))  # TODO: dupicate time cost?
            alt_seq = alt_seq.to(torch.device("cuda"))

        score_data = []  # TODO: initialize zero array and index into
        for j in range(4):
            score = []
            for model_instance in model_list[3*j:3*j+3]:
                with torch.no_grad():
                    ref = model_instance(ref_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                    alt = model_instance(alt_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                    if strand == '-':
                        ref = ref[::-1]
                        alt = alt[::-1]
                    l = 2*d+1
                    ndiff = np.abs(len(ref)-len(alt))
                    if len(ref)>len(alt):
                        alt = np.concatenate([alt[0:l//2+1],
                                              np.zeros(ndiff),
                                              alt[l//2+1:]
                                             ])  # TODO: can this be initialized?
                    elif len(ref)<len(alt):
                        alt = np.concatenate([alt[0:l//2],
                                              np.max(alt[l//2:l//2+ndiff+1], keepdims=True),
                                              alt[l//2+ndiff+1:]
                                             ])
                    score.append(alt-ref)
            score_data.append(np.mean(score, axis=0))

        score_data = np.array(score_data)
        loss = score_data[np.argmin(score_data, axis=0), np.arange(score_data.shape[1])]
        gain = score_data[np.argmax(score_data, axis=0), np.arange(score_data.shape[1])]
        return loss, gain
        
    def process_variant(self, lnum, chr, pos, ref, alt, gtf, models):
        d = self.distance
        cutoff = self.score_cutoff
        
        if ((len(set("ACGT").intersection(set(ref))) == 0) or 
            (len(set("ACGT").intersection(set(alt))) == 0) or 
            (len(ref) != 1) and 
            (len(alt) != 1) and 
            (len(ref) != len(alt))
           ):
            
            print("[Line %s]" % lnum, "WARNING, skipping variant: Variant format not supported.")
            return -1
        elif (len(ref) > (2 * d)):
            print("[Line %s]" % lnum, "WARNING, skipping variant: Deletion too large")
            return -1

        fasta = pyfastx.Fasta(self.reference_file)
        # try to make vcf chromosomes compatible with reference chromosomes
        if chr not in fasta.keys() and "chr"+chr in fasta.keys():
            chr = "chr"+chr
        elif chr not in fasta.keys() and chr[3:] in fasta.keys():
            chr = chr[3:]

        try:
            seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq
        except Exception as e:
            print(e)
            #print("[Line %s]" % lnum, "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
            #                          "See error message above.")
            print(f"[Line {lnum}]")
            print("WARNING, skipping variant:")
            print("Could not get sequence, possibly because the variant")
            print("is too close to chromosome ends. See error message above.")
            return -1

        if seq[5000+d:5000+d+len(ref)] != ref:
            #print("[Line %s]" % lnum, "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
            #      % (seq[5000+d:5000+d+len(ref)], ref))
            print("f[Line {lnum}]")
            print("WARNING, skipping variant:")
            print(f"Mismatch between FASTA (ref base: {seq[5000+d:5000+d+len(ref)]})")
            print(f"and variant file (ref base: {ref}).")
            return -1

        ref_seq = seq
        alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

        # get genes that intersect variant
        genes_pos, genes_neg = self.get_genes(chr, pos, gtf)
        if len(genes_pos)+len(genes_neg)==0:
            #print("[Line %s]" % lnum, "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
            print(f"[Line {lnum}]")
            print("WARNING, skipping variant:")
            print("Variant not contained in a gene body.")
            print("Do GTF/FASTA chromosome names match?")
            return -1

        # get splice scores
        loss_pos, gain_pos = None, None
        if len(genes_pos) > 0:
            loss_pos, gain_pos = self.compute_score(ref_seq, alt_seq, '+', d, models)
        loss_neg, gain_neg = None, None
        if len(genes_neg) > 0:
            loss_neg, gain_neg = self.compute_score(ref_seq, alt_seq, '-', d, models)

        scores = ""
        for (genes, loss, gain) in \
                ((genes_pos,loss_pos,gain_pos),(genes_neg,loss_neg,gain_neg)):
            for gene, positions in genes.items():
                warnings = "Warnings:"
                positions = np.array(positions)
                positions = positions - (pos - d)

                if self.mask == "True" and len(positions) != 0:
                    positions_filt = positions[(positions>=0) & (positions<len(loss))]
                    # set splice gain at annotated sites to 0
                    gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                    # set splice loss at unannotated sites to 0
                    not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                    loss[not_positions] = np.maximum(loss[not_positions], 0)

                elif self.mask == "True":
                    warnings += "NoAnnotatedSitesToMaskForThisGene"
                    loss[:] = np.maximum(loss[:], 0)

                if self.score_exons == "True":
                    scores1 = gene+'_sites1|'
                    scores2 = gene+'_sites2|'

                    for i in range(len(positions)//2):
                        p1, p2 = positions[2*i], positions[2*i+1]
                        if p1<0 or p1>=len(loss):
                            s1 = "NA"
                        else:
                            s1 = [loss[p1],gain[p1]]
                            s1 = round(s1[np.argmax(np.abs(s1))],2)
                        if p2<0 or p2>=len(loss):
                            s2 = "NA"
                        else:
                            s2 = [loss[p2],gain[p2]]
                            s2 = round(s2[np.argmax(np.abs(s2))],2)
                        if s1 == "NA" and s2 == "NA":
                            continue
                        scores1 += "%s:%s|" % (p1-d, s1)
                        scores2 += "%s:%s|" % (p2-d, s2)
                    scores = scores+scores1+scores2

                elif cutoff != None:
                    scores = scores+gene+'|'
                    l, g = np.where(loss<=-cutoff)[0], np.where(gain>=cutoff)[0]
                    for p, s in zip(np.concatenate([g-d,l-d]), np.concatenate([gain[g],loss[l]])):
                        scores += "%s:%s|" % (p, round(s,2))
                else:
                    scores = scores+gene+'|'
                    l, g = np.argmin(loss), np.argmax(gain),
                    scores += "%s:%s|%s:%s|" % (g-d, round(gain[g],2), l-d, round(loss[l],2))

                scores += warnings

        return scores.strip('|')

    def process(self):
        """Process a data file"""
        if self._variant_file.endswith(".vcf"):
            self.process_vcf()
        elif variants.endswith(".csv"):
            self.process_csv()
        else:
            print("ERROR, variant_file needs to be a CSV or VCF.")
            
    def process_vcf(self):
        """Process VCF file"""
        lnum = 0
        # count the number of header lines
        for line in open(self._variant_file, 'r'):
            lnum += 1
            if line[0] != '#':
                break

        variants = vcf.Reader(filename=self._variant_file)
        variants.infos["Pangolin"] = vcf.parser._Info(
            "Pangolin",'.',"String","Pangolin splice scores. "
            "Format: gene|pos:score_change|pos:score_change|...",'.','.')
        fout = vcf.Writer(open(self._output_file + ".vcf", 'w'), variants)

        for i, variant in enumerate(variants):
            scores = self.process_variant(lnum+i, 
                                     str(variant.CHROM), 
                                     int(variant.POS), 
                                     variant.REF, 
                                     str(variant.ALT[0]), 
                                     self.gtf,
                                     self.model_list)
            
            if scores != -1:
                variant.INFO["Pangolin"] = scores
            fout.write_record(variant)
            fout.flush()

        fout.close()
        
    def process_csv(self):
        """Process a CSV file"""
        col_ids = self.column_ids.split(',')
        variants = pd.read_csv(self._variant_file, header=0)
        fout = open(self._output_file + ".csv", 'w')
        fout.write(','.join(variants.columns)+',Pangolin\n')
        fout.flush()

        for lnum, variant in variants.iterrows():
            chr, pos, ref, alt = variant[col_ids]
            ref, alt = ref.upper(), alt.upper()
            scores = self.process_variant(lnum+1, 
                                     str(chr), 
                                     int(pos), 
                                     ref, 
                                     alt, 
                                     self.gtf, 
                                     self.model_list)
            
            data_out = variant.to_csv(header=False, index=False)
            data_out = data_out.split('\n')
            data_out = ','.join(data_out)
            
            if scores == -1:
                fout.write(data_out + '\n')
            else:
                fout.write(data_out + scores + '\n')
            fout.flush()

        fout.close()
