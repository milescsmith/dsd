from dsd import clean_scrnaseq
from pathlib import Path

sample_matrix=Path.home().joinpath("workspace", "dsd", "tests/data/5k_Human_PBMC_TotalSeqC_5p_nextgem_5k_Human_PBMC_TotalSeqC_5p_nextgem_count_sample_filtered_feature_bc_matrix.h5")
raw_sample_matrix=Path.home().joinpath("workspace", "dsd", "tests/data/5k_Human_PBMC_TotalSeqC_5p_nextgem_Multiplex_count_raw_feature_bc_matrix.h5")

clean_scrnaseq(sample_matrix, raw_sample_matrix)
