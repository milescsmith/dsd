import contextlib
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scar
import scvi
import torch
from anndata import ImplicitModificationWarning
from loguru import logger
from tenacity import RetryError, Retrying, stop_after_attempt


@logger.catch
def clean_scrnaseq(
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
            gex = adata[:, adata.var["feature_types"] == "Gene Exp1 ression"].copy()
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
