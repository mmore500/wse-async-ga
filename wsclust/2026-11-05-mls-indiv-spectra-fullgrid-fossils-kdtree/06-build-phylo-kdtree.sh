#!/usr/bin/env bash

set -euo pipefail

LOG_ALL="$(mktemp)"
LOG_ERR="$(mktemp)"
LOG_OUT="$(mktemp)"

# adapted from https://unix.stackexchange.com/a/61936/605206
exec > >(tee >(tee "${LOG_ALL}" >>"${LOG_OUT}")) \
     2> >(tee >(tee -a "${LOG_ALL}" >>"${LOG_ERR}") >&2)
on_exit() {
    echo
    echo "exit trap ----------------------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"

    LOGDIR="${HOME}/log/wse-async-ga/"
    echo "copying script and logs to LOGDIR ${LOGDIR}"
    mkdir -p "${LOGDIR}"

    cp "${FLOWDIR}/$(basename "$0")" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=script+ext=.sh" || :
    cp "${LOG_ALL}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stdall+ext=.log" || :
    cp "${LOG_ERR}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stderr+ext=.log" || :
    cp "${LOG_OUT}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stdout+ext=.log" || :

    echo "copying script and logs to RESULTDIR_STEP ${RESULTDIR_STEP}"
    cp "${FLOWDIR}/$(basename "$0")" "${RESULTDIR_STEP}/" || :
    cp "${LOG_ALL}" "${RESULTDIR_STEP}/stdall.log" || :
    cp "${LOG_ERR}" "${RESULTDIR_STEP}/stderr.log" || :
    cp "${LOG_OUT}" "${RESULTDIR_STEP}/stdout.log" || :
}
trap on_exit EXIT

FLOWDIR="$(realpath "$(dirname "$0")")"
FLOWNAME="$(basename "${FLOWDIR}")"
STEPNAME="$(basename "$0" .sh)"
WORKDIR="${FLOWDIR}/workdir"
RESULTDIR="${FLOWDIR}/resultdir"

###############################################################################
echo
echo
echo "============================================= ${FLOWNAME} :: ${STEPNAME}"
###############################################################################
source "${HOME}/.env" || true

###############################################################################
echo
echo "log context ------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
echo "date $(date '+%Y-%m-%d %H:%M:%S')"
echo "hostname $(hostname)"
echo "SECONDS ${SECONDS}"

echo "STEPNAME ${STEPNAME}"
echo "FLOWDIR ${FLOWDIR}"
echo "FLOWNAME ${FLOWNAME}"
echo "WORKDIR ${WORKDIR}"
echo "RESULTDIR ${RESULTDIR}"

echo "LOG_ERR ${LOG_ERR}"
echo "LOG_OUT ${LOG_OUT}"
echo "LOG_ALL ${LOG_ALL}"

###############################################################################
echo
echo "make step work dir -----------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
WORKDIR_STEP="${WORKDIR}/${STEPNAME}"
echo "WORKDIR_STEP ${WORKDIR_STEP}"
if [ -d "${WORKDIR_STEP}" ]; then
    echo "Clearing WORKDIR_STEP ${WORKDIR_STEP}"
    rm -rf "${WORKDIR_STEP}"
fi
mkdir -p "${WORKDIR_STEP}"

###############################################################################
echo
echo "make step result dir ---------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
RESULTDIR_STEP="${RESULTDIR}/${STEPNAME}"
echo "RESULTDIR_STEP ${RESULTDIR_STEP}"
if [ -d "${RESULTDIR_STEP}" ]; then
    echo "Clearing RESULTDIR_STEP ${RESULTDIR_STEP}"
    rm -rf "${RESULTDIR_STEP}"
fi
mkdir -p "${RESULTDIR_STEP}"

###############################################################################
echo
echo "setup venv  ------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
VENVDIR="${WORKDIR}/venv3x"
echo "VENVDIR ${VENVDIR}"

echo "creating venv"
source "${VENVDIR}/bin/activate"
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt"

###############################################################################
echo
echo "log source -------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
echo "log revision"
git -C "${FLOWDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/git-revision.txt"
echo "log remote"
git -C "${FLOWDIR}" remote -v > "${RESULTDIR_STEP}/git-remote.txt"
echo "log status"
git -C "$(git -C "${FLOWDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/git-status.txt"
echo "log diff"
git -C "${FLOWDIR}" --no-pager diff > "${RESULTDIR_STEP}/git-status.diff" || :
git -C "${FLOWDIR}" ls-files -z --others --exclude-standard | xargs -0 -I {} git -C "${FLOWDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/git-status.diff" || :

SRCDIR="${WORKDIR}/src"
echo "SRCDIR ${SRCDIR}"
echo "log revision"
git -C "${SRCDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/src-revision.txt"
echo "log remote"
git -C "${SRCDIR}" remote -v > "${RESULTDIR_STEP}/src-remote.txt"
echo "log status"
git -C "$(git -C "${SRCDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/src-status.txt"
echo "log diff"
git -C "${SRCDIR}" --no-pager diff > "${RESULTDIR_STEP}/src-status.diff" || :
git -C "${SRCDIR}" ls-files -z --others --exclude-standard | xargs -0 -I {} git -C "${SRCDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/src-status.diff" || :

###############################################################################
echo
echo "build phylogeny --------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
export APPTAINERENV_POLARS_MAX_THREADS=32
export APPTAINERENV_NUMBA_NUM_THREADS=32
echo "APPTAINERENV_POLARS_MAX_THREADS=${APPTAINERENV_POLARS_MAX_THREADS}"
echo "APPTAINERENV_NUMBA_NUM_THREADS=${APPTAINERENV_NUMBA_NUM_THREADS}"

export SINGULARITYENV_POLARS_MAX_THREADS="${APPTAINERENV_POLARS_MAX_THREADS}"
export SINGULARITYENV_NUMBA_NUM_THREADS="${APPTAINERENV_NUMBA_NUM_THREADS}"
echo "SINGULARITYENV_POLARS_MAX_THREADS=${SINGULARITYENV_POLARS_MAX_THREADS}"
echo "SINGULARITYENV_NUMBA_NUM_THREADS=${SINGULARITYENV_NUMBA_NUM_THREADS}"

# Alternate reconstruction, from fossils sampled within spatially compact
# k-d tree groups rather than uniformly within (layer, subgrid) strata.
# For each run (config), separately:
#   1. with joinem v0.19.0 (jexl.group_id_kdtree, jepl.SampleOp), filter to
#      non-extant, non-empty fossil genomes, then partition them into
#      KDTREE_NUM_GROUPS groups by k-d tree over min-max normalized row,
#      col, and fossil layer, with row and col further scaled by
#      KDTREE_SPATIAL_SCALE (i.e., spatial dims span [0, 8], layer [0, 1]);
#      choose groups in random order, taking at least
#      KDTREE_MIN_GROUPS groups, and more if needed to reach SAMPLE_BUDGET
#      genomes; sample SAMPLE_BUDGET genomes uniformly from chosen groups (or
#      all, if fewer). Keeps the group id column (group_id_kdtree) and adds
#      the total number of groups (kdtree_num_groups).
#   2. with hstrat, reconstruct one tree from sampled fossils, as in
#      04-build-phylo but without further sampling.
# hstrat containers bundle joinem v0.16.0, so sampling runs separately in the
# joinem container.
JOINEM_CONTAINER="docker://ghcr.io/mmore500/joinem:v0.19.0"
echo "JOINEM_CONTAINER ${JOINEM_CONTAINER}"
HSTRAT_CONTAINER="docker://ghcr.io/mmore500/hstrat:v1.27.0"
echo "HSTRAT_CONTAINER ${HSTRAT_CONTAINER}"

# Skip phyloframe's repeated contiguous-id and topological-sort precondition
# checks during reconstruction, for speed; tries built by surface_build_tree
# satisfy both (see hstrat.dataframe.surface_build_tree docstring).
export APPTAINERENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS=1
export APPTAINERENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS=1
export SINGULARITYENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS="${APPTAINERENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS}"
export SINGULARITYENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS="${APPTAINERENV_PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS}"
env | grep PHYLOFRAME_DANGEROUSLY

# Assign origin times as empirical Bayes posterior means, then pool any origin
# time inversions and collapse resulting zero-length branches (keeping leaves),
# per AssignOriginTimeEmpiricalBayesSurfPostprocessor docstring.
TRIE_POSTPROCESSOR='hstrat.CompoundTriePostprocessor(
    [
        hstrat.AssignOriginTimeEmpiricalBayesSurfPostprocessor(),
        hstrat.PipeCallablePostprocessor(
            pf.legacy.alifestd_pipe_unary_ops_polars,
            pf.legacy.alifestd_mark_num_mrca_leaf_pairs_polars,
            lambda df: pf.legacy.alifestd_pool_chronological_consistency_polars(
                df, weight_by="num_mrca_leaf_pairs"
            ),
            pf.legacy.alifestd_mark_leaves_polars,
            lambda df: pf.legacy.alifestd_coarsen_taxa_polars(
                df, by=["origin_time", "is_leaf"]
            ),
            lambda df: df.drop(
                "is_leaf", "is_root", "num_children",
                "num_leaves", "num_mrca_leaf_pairs", strict=False,
            ),
            pf.legacy.alifestd_assign_contiguous_ids_polars,
        ),
    ],
)'
echo "TRIE_POSTPROCESSOR ${TRIE_POSTPROCESSOR}"
SAMPLE_BUDGET=4000000
echo "SAMPLE_BUDGET ${SAMPLE_BUDGET}"
KDTREE_NUM_GROUPS=1000
echo "KDTREE_NUM_GROUPS ${KDTREE_NUM_GROUPS}"
KDTREE_MIN_GROUPS=50
echo "KDTREE_MIN_GROUPS ${KDTREE_MIN_GROUPS}"
KDTREE_SPATIAL_SCALE=8
echo "KDTREE_SPATIAL_SCALE ${KDTREE_SPATIAL_SCALE}"
NONEMPTY='~pl.col("data_hex").str.contains(r"^0+$")'
echo "NONEMPTY ${NONEMPTY}"

for CONFIG_DIR in "${WORKDIR}/03-postprocess"/*/; do
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"
    echo
    echo "====== Building phylogeny for config: ${CONFIG_NAME} ======"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"

    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    FOSSILS_SAMPLED="${CONFIG_WORKDIR}/a=fossils+what=kdtree-sampled+ext=.pqt"
    echo "FOSSILS_SAMPLED ${FOSSILS_SAMPLED}"

    find "${CONFIG_DIR}" -mindepth 1 -maxdepth 1 -type f \
        -name 'a=fossils+*.pqt' \
        | tee "${CONFIG_RESULTDIR}/fossil-files.txt" \
        | singularity exec "${JOINEM_CONTAINER}" \
        python3 -m joinem \
            "${FOSSILS_SAMPLED}" \
            --filter '~pl.col("is_extant")' \
            --filter "${NONEMPTY}" \
            --with-column 'pl.lit(filepath.split("/")[-2]).alias("config")' \
            --with-column 'pl.lit(filepath).cast(pl.Categorical).alias("file")' \
            --opipe "jepl.ChainOp(
                lambda df: df.collect().with_columns(
                    jexl.group_id_kdtree(
                        [
                            jexl.scaled_minmax(\"row\") * ${KDTREE_SPATIAL_SCALE},
                            jexl.scaled_minmax(\"col\") * ${KDTREE_SPATIAL_SCALE},
                            jexl.scaled_minmax(\"layer\"),
                        ],
                        n_group=${KDTREE_NUM_GROUPS},
                    ),
                ).with_columns(
                    pl.col(\"group_id_kdtree\").drop_nulls().n_unique()
                    .cast(pl.UInt32).alias(\"kdtree_num_groups\"),
                ),
                lambda df: df.join(
                    df.group_by(\"group_id_kdtree\")
                    .len()
                    .sort(\"group_id_kdtree\")
                    .sample(fraction=1, shuffle=True, seed=1)
                    .filter(
                        (pl.int_range(pl.len()) < ${KDTREE_MIN_GROUPS})
                        | (pl.col(\"len\").cum_sum() - pl.col(\"len\") < ${SAMPLE_BUDGET})
                    )
                    .select(\"group_id_kdtree\"),
                    on=\"group_id_kdtree\",
                    how=\"semi\",
                ),
                jepl.SampleOp(
                    pl.len().clip(upper_bound=${SAMPLE_BUDGET}), seed=1,
                ),
            )" \
            --how "diagonal_relaxed" \
            --eager-read --eager-write --string-cache \
            --write-kwarg 'compression="lz4"' \
            | tee "${CONFIG_RESULTDIR}/joinem-kdtree-sample.log"

    singularity exec "${JOINEM_CONTAINER}" python3 -c "
import polars as pl
df = pl.read_parquet('${FOSSILS_SAMPLED}')
print('num sampled genomes', len(df))
print('kdtree_num_groups', df['kdtree_num_groups'].unique().to_list())
print('num sampled groups', df['group_id_kdtree'].n_unique())
print(df.group_by('group_id_kdtree').len().describe())
" | tee "${CONFIG_RESULTDIR}/kdtree-sample-summary.txt"

    echo "${FOSSILS_SAMPLED}" \
        | singularity exec "${HSTRAT_CONTAINER}" \
        python3 -O -m hstrat.dataframe.surface_build_tree_over \
            "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
            --partition-by config \
            --partition-by subgrid \
            --trie-postprocessor "${TRIE_POSTPROCESSOR}" \
            --shuffle-over-same-T-seed 1 \
            --mp-pool-size 8 \
            --no-drop-dstream-metadata \
            --collapse-unif-freq 50 \
            --exploded-slice-size 100000 \
            --shrink-dtypes \
            --write-kwarg 'compression="lz4"' \
            --how "diagonal_relaxed" \
            --eager-read --eager-write --string-cache \
            --with-column 'pl.sum_horizontal(
                ((pl.col("data_hex").str.slice(2*b, 2).str.to_integer(base=16)
                ^ pl.col(f"flag_nand_mask_byte{b}"))
                & pl.col(f"flag_is_focal_mask_byte{b}")).bitwise_count_ones()
                for b in range(2)
            ).alias("focal_trait_count")' \
            --with-column 'pl.sum_horizontal(
                ((pl.col("data_hex").str.slice(2*b, 2).str.to_integer(base=16)
                ^ pl.col(f"flag_nand_mask_byte{b}"))
                & (~pl.col(f"flag_is_focal_mask_byte{b}"))).bitwise_count_ones()
                for b in range(2)
            ).cast(pl.UInt8).alias("nonfocal_trait_count")' \
            --with-column '*(
                (

                    (((
                        pl.col("data_hex")
                        .str.slice(B * 2, 2)
                        .str.to_integer(base=16)
                        ^ pl.col(f"flag_nand_mask_byte{B}")
                    ) & pl.lit(1 << b)) // pl.lit(1 << b))
                    * pl.when(
                        pl.col(f"flag_is_focal_mask_byte{B}")
                        & pl.lit(1 << b)
                        != 0
                    ).then(2).otherwise(1)
                ).cast(pl.UInt8).alias(alias)
                for B in range(2)
                for b in range(8)
                for alias in (f"trait_byte{B}_bit{b}", f"trait_num{B * 8 + b}")
            )' \
            | tee "${CONFIG_RESULTDIR}/surface_build_tree_over.log"
done

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${WORKDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${WORKDIR_STEP}"/*

cp -r "${WORKDIR_STEP}"/* "${RESULTDIR_STEP}"

find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/*

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
