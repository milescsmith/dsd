import contextlib
import warnings
from collections.abc import Iterable
from enum import StrEnum
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

import muon as mu
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
import scar
import scipy as sp
import scvi
import torch
from anndata import AnnData, ImplicitModificationWarning
from loguru import logger
from mudata import MuData
from muon import prot as pt
from numba import float32, float64, guvectorize, int32, int64, vectorize
from scipy.sparse import (
    issparse,
)
from scipy.stats import median_abs_deviation
from tenacity import RetryError, Retrying, stop_after_attempt

from dsd.logging import init_logger


class DoubletFilter(StrEnum):
    scrublet = "scrublet"
    vaeda = "vaeda"


def value_percentile(arr: np.ndarray) -> np.ndarray:
    # hacky way to loop over the array of counts and calculate each's quantile.
    # not sure why percentileofscore isn't already vectorized
    # and we have to use partial here because percentileofscore's function
    # signature is "iter, item" instead of "item, iter", meaning I cannot just pass
    # the array to score as the vectorized first argument
    return np.vectorize(partial(sp.stats.percentileofscore, a=arr))(score=arr)


# using the numba.vectorize decorator speeds this up about 13x
@vectorize(
    [
        float64(float64, float64, float64),
        float32(float32, float32, float32),
        int64(int64, int64, int64),
        int32(int32, int32, int32),
    ],
    nopython=True,
    fastmath=True,
)
def above_below(x: float, lower: float, upper: float) -> float:
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


@guvectorize([(float64[:, :], float64, float64, float64[:, :])], "(m,n),(),()->(m,n)")
def percentile_trim_rows(
    arr: npt.ArrayLike, lower: float = 0.10, upper: float = 0.99, res: npt.ArrayLike = None
) -> npt.ArrayLike:
    """
    Row-by-row, calculate the lower and upper percentiles and then use those to replace values that are
    below or above them, respectively

    NOTE: even though there are defaults listed here, they DO NOT WORK
    I don't yet know why numba ignores them.
    """
    for i in range(arr.shape[1]):
        lower_bounds = np.quantile(arr[:, i], lower)
        upper_bounds = np.quantile(arr[:, i], upper)
        res[:, i] = above_below(arr[:, i], lower_bounds, upper_bounds)


# stolen from https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#filtering-low-quality-cells
def is_outlier(adata, metric: str, nmads: int):
    met = adata.obs[metric]
    outlier = (met < np.median(met) - nmads * median_abs_deviation(met)) | (
        np.median(met) + nmads * median_abs_deviation(met) < met
    )
    return outlier


def find_isotype_controls(
    adata: AnnData, protein_control_pattern: str = "isotype", protein_control_pattern_regex: bool = False
) -> list[str]:
    if "Antibody Capture" not in adata.var["feature_types"].values:
        logger.warning(
            "None of the features in the passed anndata object are not listed as antibodies - are you sure you passed the correct modality?"
        )
    adata.var["control"] = adata.var_names.str.contains(protein_control_pattern, regex=protein_control_pattern_regex)
    return adata.var.loc[lambda y: y["control"]].index.to_list()


def std_quality_control(
    mdata: MuData,
    min_genes: int = 500,
    max_gene_total_counts: int | None = None,  # = 6000,
    max_gene_counts_percentile: float | None = None,  # = 0.95,
    max_protein_total_counts: int | None = None,
    max_protein_counts_percentile: float | None = None,
    min_gene_total_counts: int = 1000,
    min_n_cells_by_counts: int = 3,
    max_percent_mt: float = 10.0,
    remove_by_std_deviation: bool = False,
    protein_isotype_controls: str | Iterable[str] | None = None,
    protein_control_pattern: str = "isotype",
    protein_control_pattern_regex: bool = False,
    remove_isotype_outliers: bool = True,
    remove_all_feature_outliers: bool = False,
    max_isotype_counts_percentile: float = 95.0,
    doublet_algorithm: DoubletFilter = DoubletFilter.scrublet,
    verbose: bool = False,
):
    """Standard quality control filtering for paired scRNAseq/CITEseq sample. For use as part of creating a data standard packakage.

    Parameters
    ----------
    mdata : MuData
        MuData object to process. Should have 'rna' and 'prot' modalities.
    min_genes : int, optional
        Minimum number of detected genes necessary to keep a cell, by default 500
    max_gene_total_counts : int | None, optional
        Maximum total gene counts permitted in each cell; cells with more are removed. If "max_gene_counts_percentile" is also passed, this is ignored., by default None
    max_gene_counts_percentile : float | None, optional
        Maximum gene count percentile cutoff beyond which cells are removed, by default None
    max_protein_total_counts : int | None, optional
        Maximum total antibody counts permitted in each cell; cells with more are removed. If "max_gene_counts_percentile" is also passed, this is ignored., by default None
    max_protein_counts_percentile : float | None, optional
        Maximum antibody count percentile cutoff beyond which cells are removed, by default None
    min_gene_total_counts : int, optional
        _description_, by default 1000
    min_n_cells_by_counts : int, optional
        _description_, by default 3
    max_percent_mt : float, optional
        _description_, by default 10.0
    remove_by_std_deviation : bool, optional
        _description_, by default False
    protein_isotype_controls : str | Iterable[str] | None, optional
        A antibodies to be used as isotype controls, by default None
    protein_control_pattern : str, optional
        Instead of an explicit list of isotype controls, a pattern to be used to find isotype controls based on their name , by default "isotype"
    protein_control_pattern_regex : bool, optional
        Should "protein_control_pattern" be intrepreted as a regex pattern?, by default False
    remove_isotype_outliers : bool, optional
        Should cells with isotype counts above the "max_isotype_counts_percentile"th percentile be removed?, by default True
    max_isotype_counts_percentile : float, optional
        _description_, by default 95.0
    remove_all_feature_outliers : bool, optional
        Should cells with any counts above the 95th percentile be removed? Currently unused, by default False
    doublet_algorithm : DoubletFilter, optional
        Method to use for doublet detection. Must be either "scrublet" or "vaeda", though "vaeda" doesn't necessarily work due to current tensorflow/tf-keras incompatibilities, by default DoubletFilter.scrublet
    verbose : bool, optional
        Enable verbose logging, by default False
    """
    if verbose:
        logger.enable(__package__)
        init_logger(3)
    else:
        init_logger(0)
    mdata.var_names_make_unique()
    mdata["rna"].var["mt"] = mdata["rna"].var_names.str.startswith("MT-")
    mdata["rna"].var["ribo"] = mdata["rna"].var_names.str.startswith(("RPS", "RPL"))
    mdata["rna"].var["hb"] = mdata["rna"].var_names.str.contains(r"^HB[^(P)]", regex=True)
    if not protein_isotype_controls:
        protein_isotype_controls = find_isotype_controls(
            mdata["prot"], protein_control_pattern, protein_control_pattern_regex
        )
        logger.info(
            f"Found and using {protein_isotype_controls} as isotype controls. If that is incorrect, try passing the actual names to use."
        )
    elif protein_isotype_controls:
        protein_isotype_controls = (
            [protein_isotype_controls] if isinstance(protein_isotype_controls, str) else protein_isotype_controls
        )
        mdata["prot"].var["control"] = mdata["prot"].var["control"].isin(protein_isotype_controls)

    logger.info("Calculating gene expression QC metrics")
    sc.pp.calculate_qc_metrics(
        adata=mdata["rna"],
        qc_vars=["mt", "ribo", "hb"],
        percent_top=(20, 50, 100, 200, 500),
        log1p=True,
        inplace=True,
    )

    logger.info("Calculating antibody QC metrics")
    try:
        sc.pp.calculate_qc_metrics(
            adata=mdata["prot"],
            percent_top=(5, 10, 15),
            var_type="antibodies",
            qc_vars=("control",),
            inplace=True,
            log1p=True,
        )
    except IndexError:
        logger.error(
            "Calculating antibody QC metrics failed, probably due to too few antibodies. Retrying without calculating the top antibodies by count."
        )
        sc.pp.calculate_qc_metrics(
            adata=mdata["prot"],
            percent_top=None,
            var_type="antibodies",
            qc_vars=("control",),
            inplace=True,
            log1p=True,
        )

    logger.info("Filtering cells and genes")
    mdata["rna"].obs["outlier"] = (
        is_outlier(mdata["rna"], "log1p_total_counts", 5)
        | is_outlier(mdata["rna"], "log1p_n_genes_by_counts", 5)
        | is_outlier(mdata["rna"], "pct_counts_in_top_20_genes", 5)
        | is_outlier(mdata["rna"], "pct_counts_mt", 3)
        | (mdata["rna"].obs["pct_counts_mt"] > max_percent_mt)
    )

    logger.info("Filtering cells and antibodies")
    mdata["prot"].obs["outlier"] = is_outlier(mdata["prot"], "log1p_total_counts", 5) | is_outlier(
        mdata["prot"], "log1p_n_antibodies_by_counts", 5
    )

    if remove_by_std_deviation:
        logger.info("Removing 'outliers'")
        mu.pp.filter_obs(mdata["rna"], "outlier", lambda x: ~x)
        mu.pp.filter_obs(mdata["prot"], "outlier", lambda x: ~x)
    else:
        if max_gene_counts_percentile:
            if max_gene_total_counts:
                logger.warning(
                    "Both max_genes_quantile and max_total_counts were specified. Ignoring max_total_counts."
                )
            logger.info("Calculating gene expression count percentiles")
            mdata["rna"].obs["gene_counts_percentile"] = value_percentile(mdata["rna"].obs["total_counts"])
            logger.info("Filtering cells by gene expression count percentile")
            mu.pp.filter_obs(mdata["rna"], "gene_counts_percentile", lambda x: x < max_gene_counts_percentile)
        elif max_gene_total_counts:
            logger.info("Filtering cells by total gene expression counts")
            mu.pp.filter_obs(mdata["rna"], "total_counts", lambda x: x <= min_gene_total_counts)

        if max_protein_counts_percentile:
            if max_protein_total_counts:
                logger.warning(
                    "Both max_protein_counts_quantile and max_protein_total_counts were specified. Ignoring max_total_counts."
                )
            logger.info("Calculating antibody count percentiles")
            mdata["prot"].obs["protein_counts_percentile"] = value_percentile(mdata["prot"].obs["total_counts"])
            logger.info("Filtering cells by antibody count percentile")
            mu.pp.filter_obs(mdata["prot"], "protein_counts_percentile", lambda x: x < max_protein_counts_percentile)
        elif max_protein_total_counts:
            logger.info("Filtering cells by total antibody counts")
            mu.pp.filter_obs(mdata["prot"], "total_counts", lambda x: x < max_protein_counts_percentile)

        mu.pp.filter_var(mdata["rna"], "n_cells_by_counts", lambda x: x >= min_n_cells_by_counts)
        mu.pp.filter_obs(mdata["rna"], "n_genes_by_counts", lambda x: x >= min_genes)

    if remove_isotype_outliers:
        for isotype in protein_isotype_controls:
            logger.info(f"Calculating {isotype} percentiles")
            values = mdata["prot"][:, isotype].X.toarray() if issparse(mdata["prot"].X) else mdata["prot"][:, isotype].X
            mdata["prot"].obs[f"{isotype} percentile"] = value_percentile(values.flatten())

        mu.pp.intersect_obs(mdata)
        for isotype in protein_isotype_controls:
            logger.info(f"Removing {isotype} outliers")
            # do this twice because if it is all in the same loop, the quantile
            # calculations are affected by the removal of the first noisy samples
            mu.pp.filter_obs(
                mdata,
                f"prot:{isotype} percentile",
                lambda x: x < max_isotype_counts_percentile,
            )

    match doublet_algorithm:
        case DoubletFilter.scrublet:
            logger.info("Running Scrublet")
            sc.pp.scrublet(mdata["rna"])
            mu.pp.filter_obs(mdata["rna"], "predicted_doublet", lambda x: ~x)
        case DoubletFilter.vaeda:
            try:
                import vaeda
            except ImportError:
                msg = "Cannot import vaeda"
                logger.error(msg, exc_info=True)
            logger.info("Running vaeda")
            adata = vaeda.vaeda(mdata["rna"].copy(), seed=20150318)
            mdata["rna"].obs = mdata["rna"].obs.join(adata.obs[["vaeda_scores", "vaeda_calls"]], how="left")
            mu.pp.filter_obs(mdata["rna"], "vaeda_calls", lambda x: x == "singlet")

    mu.pp.intersect_obs(mdata)
    logger.info("Finished QC")


def std_process(
    filtered: MuData,
    raw: MuData | None = None,
    protein_isotype_controls: str | Iterable[str] | None = None,
    protein_control_pattern: str = "isotype",
    protein_control_pattern_regex: bool = False,
):
    if not protein_isotype_controls:
        if "control" not in filtered["prot"].var.columns:
            logger.error("Strongly suggest running std_quality_control() first.")
        protein_isotype_controls = find_isotype_controls(
            filtered["prot"], protein_control_pattern, protein_control_pattern_regex
        )

    filtered["prot"].layers["counts"] = filtered["prot"].X.copy()
    if raw is not None:
        logger.info("running dsb")
        pt.pp.dsb(
            data=filtered,
            data_raw=raw,
            isotype_controls=protein_isotype_controls,
            add_layer=False,
        )

    logger.info("Normalizing")
    sc.pp.normalize_total(adata=filtered["rna"], target_sum=1e4)

    logger.info("Log-transforming")
    sc.pp.log1p(filtered["rna"])

    logger.info("Finding variable genes")
    sc.pp.highly_variable_genes(filtered["rna"], min_mean=0.0125, max_mean=3, min_disp=0.5)

    filtered["rna"].raw = filtered["rna"]

    logger.info("Scaling data")
    sc.pp.scale(filtered["rna"], max_value=10)


@logger.catch
def scvi_clean_scrnaseq(
    sample_matrix: Path | str,
    raw_sample_matrix: Path | str,
    device: Literal["cuda", "mps", "cpu"] | None = None,
    gex_scar_prob_cutoff: float = 0.995,
    gex_scar_prob_retry_cutoff_steps: tuple[int] | list[int] | None = None,
    prot_scar_prob_cutoff: float = 0.995,
    prot_scar_prob_retry_cutoff_steps: tuple[int] | list[int] | None = None,
    min_genes: int = 500,
    max_genes_quantile: float = 0.95,
    min_gene_total_counts: int = 1000,
    min_cells: int = 3,
    max_percent_mt: float = 0.1,
    output_path: Path | str | None = None,
    num_cpus: int | None = None,
    skip_make_unique: bool = False,
) -> None:
    sample_matrix = Path(sample_matrix) if isinstance(sample_matrix, str) else sample_matrix
    raw_sample_matrix = Path(raw_sample_matrix) if isinstance(raw_sample_matrix, str) else raw_sample_matrix
    if not num_cpus:
        num_cpus = cpu_count() - 1

    scvi.settings.dl_num_workers = num_cpus
    if not output_path:
        output_path = sample_matrix.parent.resolve()
    elif isinstance(output_path, str):
        output_path = Path(output_path)

    # if you see some warning about variable names not being unique,
    # you might be tempted to try to "fix" this by running `adata.obs_names_make_unique()` and
    # `adata.var_names_make_unique(); you would, however, be wrong to do so.
    # There's a good chance the warning is because there is an antibody and gene that share a name.
    # If you run `var_names_make_unique()`, it will append a `-1` to one of those names, and then after slicing
    # and running `scar.setup_anndata` and `scar.model.train`, you'll encounter a `tensor_a must be the same shape as
    # tensor_b` error - this is because scar doesn't know how to handle proteins with the suffix and has
    # incorrectly generated the ambient_profile -- the prot.uns["ambient_profile_all"] and
    # prot.uns["ambient_profile_Antibody Capture"] do not match adata.var_names, and things get messed up.
    with warnings.catch_warnings():
        logger.debug("Loading filtered sample_matrix")
        warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
        adata = sc.read_10x_h5(sample_matrix, gex_only=False)
        logger.debug("Loading raw sample_matrix")
        unfiltered_adata = sc.read_10x_h5(raw_sample_matrix, gex_only=False)

        if "Gene Expression" in adata.var["feature_types"].values:
            logger.debug("Gene expression modality found")
            rnaseq = True
            gex = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            gex.obs_names_make_unique()
            gex.var_names_make_unique()
            unfiltered_gex = unfiltered_adata[:, unfiltered_adata.var["feature_types"] == "Gene Expression"].copy()
            unfiltered_gex.obs_names_make_unique()
            unfiltered_gex.var_names_make_unique()
        else:
            logger.debug("No gene expression data found. Skipping.")
            rnaseq = False

        if "Antibody Capture" in adata.var["feature_types"].values:
            logger.debug("Antibody capture modality found")
            citeseq = True
            prot = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
            unfiltered_prot = unfiltered_adata[:, unfiltered_adata.var["feature_types"] == "Antibody Capture"].copy()
        else:
            logger.debug("No antibody data found. Skipping.")
            citeseq = False

    if device is None:
        match (torch.cuda.is_available(), torch.backends.mps.is_available()):
            case (True, False) | (True, True):
                device = "cuda"
            case (False, True):
                device = "mps"
            case (False, False):
                device = "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    logger.debug(f"Using {device} to generate models")

    if rnaseq:
        logger.debug("Processing RNAseq data")
        if gex_scar_prob_retry_cutoff_steps is None:
            gex_scar_prob_retry_cutoff_steps = np.linspace(0.0, 0.1, num=9)
        gex_prob_cutoff = [gex_scar_prob_cutoff - i for i in gex_scar_prob_retry_cutoff_steps]
        logger.debug(f"using {gex_prob_cutoff} scar probabilities")

        with contextlib.suppress(RetryError):
            logger.info("Setting up gene expression AnnData for scAR")
            for attempt in Retrying(stop=stop_after_attempt(len(gex_prob_cutoff))):
                with attempt:
                    prob = gex_prob_cutoff[attempt.retry_state.attempt_number - 1]
                    logger.debug(f"Trying {prob=}")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ImplicitModificationWarning, module="scar")
                        scar.setup_anndata(
                            adata=gex,
                            raw_adata=unfiltered_gex,
                            feature_type="Gene Expression",
                            prob=prob,
                            kneeplot=False,
                        )
            logger.debug(f"Succeeded with {prob=}")
        if attempt.retry_state.outcome.failed:
            msg = "All attempts to calculate the probability of each gene having ambient RNA failed. Try adjusting `gex_scar_prob_retry_cutoff_steps`"
            raise ValueError(msg)

        logger.info("Generating ambient RNA model")
        gex_scar = scar.model(
            raw_count=gex,
            ambient_profile=gex.uns["ambient_profile_Gene Expression"],
            feature_type="mRNA",
            sparsity=1,
            device=device,  # CPU, CUDA and MPS are supported.
        )

        logger.debug("Finished model generation, training model")
        gex_scar.train(epochs=200, batch_size=32, verbose=True)

        logger.debug("Inferring ambient RNA noise")
        gex_scar.inference()
        gex.layers["scar_denoised"] = gex_scar.native_counts.copy()
        logger.debug("Finished inferring noise with scAR")

        logger.debug("Setting up gene expression AnnData for SOLO")
        scvi.model.SCVI.setup_anndata(gex)
        logger.info("Training scVI gex model")
        vae = scvi.model.SCVI(gex)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="multiprocessing")
            warnings.filterwarnings("ignore", category=UserWarning, module="scvi")
            vae.train(early_stopping=True, accelerator=device, load_sparse_tensor=True, check_val_every_n_epoch=1)
            logger.debug("Converting scVI model to SOLO")
            solo = scvi.external.SOLO.from_scvi_model(vae)
            logger.debug("Training SOLO model")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="multiprocessing")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            solo.train(early_stopping=True)
            logger.debug("Predicting singlets/doublet probabilities and transferring noise and singlet probabilites")
            gex.obs["solo_call"] = solo.predict(soft=False)
            gex.obs = pd.merge(gex.obs, solo.predict(return_logits=True), left_index=True, right_index=True)

        logger.debug("Subsetting")
        filtered_gex = gex[gex.obs["solo_call"] == "singlet", :].copy()
        filtered_gex.layers["pre_scar_counts"] = filtered_gex.X.copy()
        filtered_gex.X = filtered_gex.layers["scar_denoised"]

        logger.debug("Performing QC on denoised singlet data")

        filtered_gex.var["mt"] = filtered_gex.var_names.str.startswith("MT-")

        sc.pp.calculate_qc_metrics(
            adata=filtered_gex,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        logger.debug(f"Removing genes present in < {min_cells} cells")
        sc.pp.filter_genes(filtered_gex, min_cells=min_cells)
        logger.debug(f"Removing cells with < {min_genes} genes")
        sc.pp.filter_cells(filtered_gex, min_genes=min_genes)
        logger.debug(f"Removing cells with < {min_gene_total_counts} gene counts")
        sc.pp.filter_cells(filtered_gex, min_counts=min_gene_total_counts)
        logger.debug(f"Removing cells with > {max_percent_mt} mitochondrial genes")
        filtered_gex = filtered_gex[filtered_gex.obs.pct_counts_mt < max_percent_mt * 100, :].copy()

        filtered_gex.raw = gex

        logger.info("Finished processing gene expression data")

        gex_output_file = output_path.joinpath("solo_filtered_scar_denoised_scrnaseq.h5ad")
        logger.info(f"Writing data to {gex_output_file.resolve()}")
        filtered_gex.write(gex_output_file)

    if citeseq:
        logger.debug("Processing CITEseq data")
        if prot_scar_prob_retry_cutoff_steps is None:
            prot_scar_prob_retry_cutoff_steps = np.linspace(0.0, 0.5, num=17)
        prot_prob_cutoff = [prot_scar_prob_cutoff - i for i in prot_scar_prob_retry_cutoff_steps]
        logger.debug(f"using {prot_prob_cutoff} scar probabilities")
        with contextlib.suppress(RetryError):
            for attempt in Retrying(stop=stop_after_attempt(len(prot_prob_cutoff))):
                with attempt:
                    prob = prot_prob_cutoff[attempt.retry_state.attempt_number - 1]
                    logger.debug(f"Trying {prob=}")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ImplicitModificationWarning, module="scar")
                        scar.setup_anndata(
                            adata=prot,
                            raw_adata=unfiltered_prot,
                            feature_type="Antibody Capture",
                            prob=prob,
                            kneeplot=False,
                        )
            logger.debug(f"Succeeded with {prob=}")
        if attempt.retry_state.outcome.failed:
            msg = "All attempts to calculate the probability of each protein having ambient reads failed. Try adjusting `prot_scar_prob_retry_cutoff_steps`"
            raise ValueError(msg)

        logger.info("Generating ambient antibody model")
        prot_scar = scar.model(
            raw_count=prot,  # In the case of Anndata object, scar will automatically use the estimated ambient_profile present in adata.uns.
            ambient_profile=prot.uns["ambient_profile_Antibody Capture"],
            feature_type="ADT",
            count_model="binomial",
            device=device,
        )

        logger.debug("Finished model generation, training model")
        # TODO: make a batch_size option or retry loop
        prot_scar.train(epochs=500, verbose=True)  # , batch_size=32, verbose=True)

        logger.debug("Inferring ambient antibody noise")
        prot_scar.inference()
        prot_scar.native_counts.toarray()
        prot.layers["scar_denoised"] = prot_scar.native_counts.copy()
        logger.debug("Finished inferring noise with scAR")

        logger.debug("Transferring noise probabilities")
        # filtered_prot = prot[filtered_gex.obs_names, :]
        filtered_prot = prot
        filtered_prot.layers["pre_scar_counts"] = filtered_prot.X.copy()
        filtered_prot.X = filtered_prot.layers["scar_denoised"].astype("float64")

        logger.info("Finished processing antibody expression data")

        antibody_output_file = output_path.joinpath("solo_filtered_scar_denoised_citeseq.h5ad")
        filtered_prot.raw = prot

        logger.info(f"Writing data to {antibody_output_file.resolve()}")

        filtered_prot.write(antibody_output_file)
