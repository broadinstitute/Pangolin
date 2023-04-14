import gffutils


def create_db(annotation_file, filter_str=None):
    """Create an output database
    
    Parameters
    ----------
    annotation_file : str, filepath for db file to be made
    filter_str : str, or None. Format: tag1,tag2,... 
        or None to keep all features. 
        Default: Ensembl_canonical
    Returns
    -------
    gffutils.create_db output
    """
    if annotation_file.endswith(".gtf"):
        prefix = annotation_file[:-4]
    elif annotation_file.endswith(".gtf.gz"):
        prefix = annotation_file[:-7]
    else:
        print(f"ERROR, annotation_file {annotation_file} should be a GTF file.")

    def transform_filter(feat):
        if feat.featuretype not in ["gene", "transcript", "exon"]:
            return False
        elif filter_str != "None" and feat.featuretype in ["transcript", "exon"]:
            present = False
            for tag in filter_str.split(','):
                if "tag" in feat.attributes and tag in feat["tag"]:
                    present = True
            if not present:
                return False
        return feat

    db = gffutils.create_db(annotation_file, 
                            prefix+".db", 
                            force=True,
                            disable_infer_genes=True,
                            disable_infer_transcripts=True,
                            transform=transform_filter)
    return db
