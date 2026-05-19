import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numba as nb
import numpy as np
from numba import cuda

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import mix3r_int_weights as base


THREADS_PER_BLOCK = 128
CPU_WORKERS_OVERRIDE = None
_GPU_CACHE = {}
BIVARIATE_MAXITER_MULTIPLIER = 4


def get_cpu_worker_count(config, n_jobs, phase=None):
    phase_key = None if phase is None else phase.upper()
    requested = None

    if isinstance(CPU_WORKERS_OVERRIDE, dict):
        if phase is not None:
            requested = CPU_WORKERS_OVERRIDE.get(phase)
        if requested is None:
            requested = CPU_WORKERS_OVERRIDE.get("default")
    elif CPU_WORKERS_OVERRIDE is not None:
        requested = CPU_WORKERS_OVERRIDE

    if requested is None and phase is not None:
        requested = config.get(f"cpu_workers_{phase}")
    if requested is None:
        requested = config.get("cpu_workers")

    env_names = []
    if phase_key is not None:
        env_names.extend(
            (
                f"MIX3R_INT_WEIGHTS_THREADS_{phase_key}",
                f"APPTAINERENV_MIX3R_INT_WEIGHTS_THREADS_{phase_key}",
            )
        )
    env_names.extend(
        (
            "MIX3R_INT_WEIGHTS_THREADS",
            "APPTAINERENV_MIX3R_INT_WEIGHTS_THREADS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
        )
    )
    if requested is None:
        for env_name in env_names:
            env_value = os.environ.get(env_name)
            if env_value:
                requested = int(env_value)
                break

    if requested is None:
        requested = os.cpu_count() or 1
    return max(1, min(int(requested), int(n_jobs)))


def get_bivariate_optimization_iters(config):
    optimization = config["optimization"]
    multiplier = int(optimization.get("maxiter_2d_multiplier", BIVARIATE_MAXITER_MULTIPLIER))
    base_glob = int(optimization["maxiter_2d_glob"])
    base_loc = int(optimization["maxiter_2d_loc"])
    maxiter_2d_glob = int(optimization.get("maxiter_2d_glob_optimized", base_glob * multiplier))
    maxiter_2d_loc = int(optimization.get("maxiter_2d_loc_optimized", base_loc * multiplier))

    print(
        "Bivariate optimization iterations: "
        f"global={maxiter_2d_glob}, local={maxiter_2d_loc} "
        f"(base global={base_glob}, base local={base_loc})",
        flush=True,
    )
    return maxiter_2d_glob, maxiter_2d_loc


def get_chrom_ranges(snps_df):
    chr_values = snps_df.CHR.to_numpy()
    if chr_values.size == 0:
        return []
    boundaries = np.flatnonzero(chr_values[1:] != chr_values[:-1]) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    stops = np.concatenate((boundaries, np.array([chr_values.size])))
    return [(int(chr_values[start]), int(start), int(stop)) for start, stop in zip(starts, stops)]


def _prune_one_chrom(job):
    chrom, start, stop, template_dir, ld_n_all, b2keep_all, rand_pval, r2_prune_thresh = job
    ld_r2_file = os.path.join(template_dir, f"chr{chrom}.ld_r2")
    ld_idx_file = os.path.join(template_dir, f"chr{chrom}.ld_idx")
    r2 = np.memmap(ld_r2_file, dtype="f4", mode="r")
    r2_idx = np.memmap(ld_idx_file, dtype="i4", mode="r")
    ld_n = ld_n_all[start:stop]
    b2use = b2keep_all[start:stop]
    bpruned = base.prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
    mean_ld = float(ld_n[bpruned].mean()) if bpruned.any() else float("nan")
    return chrom, start, stop, bpruned, int(bpruned.sum()), mean_ld


def select_snps(snps_df, *, snps2keep, n_random, do_pruning, r2_prune_thresh, template_dir, rng_seed):
    print("Selecting variants.")
    b2keep = snps_df.IS_VALID.to_numpy(copy=True)
    print(f"    {b2keep.sum()} remains after applying MAF, Z and INFO filters.")
    rng = np.random.default_rng(seed=rng_seed)
    if snps2keep:
        b2keep &= snps_df.SNP.isin(snps2keep).to_numpy()
        print(f"    {b2keep.sum()} remains after restricting to provided list of variants.")
    if do_pruning:
        print("    Performing pruning.")
        chrom_ranges = get_chrom_ranges(snps_df)
        ld_n_all = snps_df.LD_N.to_numpy()
        rand_pvals = [
            rng.random(stop - start)
            for _, start, stop in chrom_ranges
        ]
        workers = get_cpu_worker_count({}, len(chrom_ranges), phase="pruning")
        print(f"    Using {workers} CPU worker(s) for chromosome pruning.")
        jobs = [
            (chrom, start, stop, template_dir, ld_n_all, b2keep, rand_pvals[i], r2_prune_thresh)
            for i, (chrom, start, stop) in enumerate(chrom_ranges)
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chrom, start, stop, bpruned, n_survived, mean_ld in executor.map(_prune_one_chrom, jobs):
                b2keep[start:stop] = bpruned
                print(f"        {n_survived} SNPs survive pruning on chromosome {chrom}")
                print(f"            {mean_ld:.2f} mean size of LD block of pruned SNPs")
        print(f"    {b2keep.sum()} SNPs remain in total after pruning.")
    if n_random:
        i2keep = np.flatnonzero(b2keep)
        assert n_random <= len(i2keep)
        i2keep = rng.choice(i2keep, n_random, replace=False)
        b2keep[:] = False
        b2keep[i2keep] = True
        print(f"    {b2keep.sum()} SNPs remain after taking a random subset.")
    return snps_df.SNP[b2keep].to_numpy(copy=True)


def _load_opt_data_one_chrom(job):
    chrom, start, stop, template_dir, nbin_het_hist, selected_mask, z_cols, n_cols, snps_df = job
    ld_r2_file = os.path.join(template_dir, f"chr{chrom}.ld_r2")
    ld_idx_file = os.path.join(template_dir, f"chr{chrom}.ld_idx")
    r2 = np.memmap(ld_r2_file, dtype="f4", mode="r")
    r2_idx = np.memmap(ld_idx_file, dtype="i4", mode="r")
    snps_df_chr = snps_df.iloc[start:stop]
    het = 2 * snps_df_chr.MAF.to_numpy() * (1 - snps_df_chr.MAF.to_numpy())
    ld_n = snps_df_chr.LD_N.to_numpy()
    b2keep = selected_mask[start:stop]
    r2_het = base.get_r2_het(r2, r2_idx, het)
    r2_het_hist = base.get_r2_het_hist(b2keep, r2_het, ld_n, nbin_het_hist)
    ld_scores = base.get_ld_scores(b2keep, r2_het, ld_n, nbin_het_hist)

    out = {}
    for z_col, n_col in zip(z_cols, n_cols):
        out[z_col] = snps_df_chr.loc[b2keep, z_col].to_numpy()
        out[n_col] = snps_df_chr.loc[b2keep, n_col].to_numpy()
    return chrom, r2_het_hist, ld_scores, out


def load_opt_data(template_dir, snps_df, *, snps2keep, nbin_het_hist):
    print("Loading LD data")
    z_cols = [c for c in sorted(snps_df.columns) if c.startswith("Z_")]
    n_cols = [c for c in sorted(snps_df.columns) if c.startswith("N_")]
    assert all(z_col.split("_")[1] == n_col.split("_")[1] for z_col, n_col in zip(z_cols, n_cols))
    selected_mask = snps_df.SNP.isin(snps2keep).to_numpy()
    chrom_ranges = get_chrom_ranges(snps_df)
    workers = get_cpu_worker_count({}, len(chrom_ranges), phase="ld_prep")
    print("Processing chromosomes: ", end="")
    for chrom, _, _ in chrom_ranges:
        print(f"{chrom} ", end="")
    print()
    print(f"Using {workers} CPU worker(s) for LD preparation.")

    jobs = [
        (chrom, start, stop, template_dir, nbin_het_hist, selected_mask, z_cols, n_cols, snps_df)
        for chrom, start, stop in chrom_ranges
    ]
    r2_het_hist_list = []
    ld_scores_list = []
    z_n_dict = {c: [] for c in z_cols + n_cols}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, r2_het_hist, ld_scores, out in executor.map(_load_opt_data_one_chrom, jobs):
            r2_het_hist_list.append(r2_het_hist)
            ld_scores_list.append(ld_scores)
            for col, values in out.items():
                z_n_dict[col].append(values)

    r2_het_hist = np.concatenate(r2_het_hist_list)
    ld_scores = np.concatenate(ld_scores_list)
    for col, val_list in z_n_dict.items():
        z_n_dict[col] = np.concatenate(val_list)
    print(f"{z_n_dict['Z_0'].size} SNPs loaded")
    print(f"{r2_het_hist.sum()/z_n_dict['Z_0'].size:.2f} mean size of LD block of loaded SNPs")
    return r2_het_hist, z_n_dict, ld_scores


def _total_het_one_chrom(job):
    chrom, start, stop, template_dir, rand_pval, r2_prune_thresh, is_valid, maf, ld_n_all = job
    ld_r2_file = os.path.join(template_dir, f"chr{chrom}.ld_r2")
    ld_idx_file = os.path.join(template_dir, f"chr{chrom}.ld_idx")
    r2 = np.memmap(ld_r2_file, dtype="f4", mode="r")
    r2_idx = np.memmap(ld_idx_file, dtype="i4", mode="r")
    het = 2 * maf[start:stop] * (1 - maf[start:stop])
    ld_n = ld_n_all[start:stop]
    b2use = is_valid[start:stop]
    bpruned = base.prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
    return base.get_total_het_used_chr(bpruned, r2_idx, het, ld_n)


def get_total_het_used(template_dir, snps_df, rand_prune_seed, r2_prune_thresh):
    rng = np.random.default_rng(rand_prune_seed)
    chrom_ranges = get_chrom_ranges(snps_df)
    is_valid = snps_df.IS_VALID.to_numpy()
    maf = snps_df.MAF.to_numpy()
    ld_n_all = snps_df.LD_N.to_numpy()
    rand_pvals = [rng.random(stop - start) for _, start, stop in chrom_ranges]
    workers = get_cpu_worker_count({}, len(chrom_ranges), phase="total_het")
    print(f"Using {workers} CPU worker(s) for total heterozygosity calculation.")
    jobs = [
        (chrom, start, stop, template_dir, rand_pvals[i], r2_prune_thresh, is_valid, maf, ld_n_all)
        for i, (chrom, start, stop) in enumerate(chrom_ranges)
    ]
    total_het = 0.0
    total_n = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for total_het_chr, total_n_chr in executor.map(_total_het_one_chrom, jobs):
            total_het += total_het_chr
            total_n += total_n_chr
    return len(snps_df) * total_het / total_n


class GPUCostCache:
    def __init__(self, *, dtype, nbin_r2_het_hist, z_arrays, n_arrays, r2_het_hist, ld_scores):
        self.dtype = dtype
        self.nbin_r2_het_hist = nbin_r2_het_hist
        self.z_gpu = [cuda.to_device(np.ascontiguousarray(arr, dtype=dtype)) for arr in z_arrays]
        self.n_gpu = [cuda.to_device(np.ascontiguousarray(arr, dtype=dtype)) for arr in n_arrays]
        self.r2_het_hist_gpu = cuda.to_device(np.ascontiguousarray(r2_het_hist))
        self.ld_scores_gpu = cuda.to_device(np.ascontiguousarray(ld_scores))
        self.res_gpu = cuda.device_array(shape=z_arrays[0].shape, dtype=dtype)
        self.res_host = np.empty(z_arrays[0].shape, dtype=dtype)
        self.blocks_per_grid = math.ceil(z_arrays[0].shape[0] / THREADS_PER_BLOCK)

    def sum_result(self):
        self.res_gpu.copy_to_host(self.res_host)
        return float(self.res_host.sum(dtype=self.dtype))


def _cache_key(tag, *arrays_and_meta):
    key = [tag]
    for item in arrays_and_meta:
        if isinstance(item, np.ndarray):
            key.extend((id(item), item.shape, item.dtype.str))
        else:
            key.append(item)
    return tuple(key)


def get_gpu_cache_1d(z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    key = _cache_key("1d", z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    cache = _GPU_CACHE.get(key)
    if cache is None:
        cache = GPUCostCache(
            dtype=np.float32,
            nbin_r2_het_hist=nbin_r2_het_hist,
            z_arrays=[z0_vec],
            n_arrays=[n_vec],
            r2_het_hist=r2_het_hist,
            ld_scores=ld_scores,
        )
        _GPU_CACHE[key] = cache
    return cache


def get_gpu_cache_2d(z0_1_vec, z0_2_vec, n_1_vec, n_2_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    key = _cache_key("2d", z0_1_vec, z0_2_vec, n_1_vec, n_2_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    cache = _GPU_CACHE.get(key)
    if cache is None:
        cache = GPUCostCache(
            dtype=np.float32,
            nbin_r2_het_hist=nbin_r2_het_hist,
            z_arrays=[z0_1_vec, z0_2_vec],
            n_arrays=[n_1_vec, n_2_vec],
            r2_het_hist=r2_het_hist,
            ld_scores=ld_scores,
        )
        _GPU_CACHE[key] = cache
    return cache


def get_gpu_cache_3d(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    key = _cache_key("3d", z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    cache = _GPU_CACHE.get(key)
    if cache is None:
        cache = GPUCostCache(
            dtype=np.float64,
            nbin_r2_het_hist=nbin_r2_het_hist,
            z_arrays=[z0_1_vec, z0_2_vec, z0_3_vec],
            n_arrays=[n_1_vec, n_2_vec, n_3_vec],
            r2_het_hist=r2_het_hist,
            ld_scores=ld_scores,
        )
        _GPU_CACHE[key] = cache
    return cache


def cost_1d_gpu(z0_vec, p, sb2, s02, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    p, sb2, s02 = map(nb.float32, (p, sb2, s02))
    cache = get_gpu_cache_1d(z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    base.log_pdf_1d[cache.blocks_per_grid, THREADS_PER_BLOCK](
        cache.res_gpu,
        cache.z_gpu[0],
        p,
        sb2,
        s02,
        cache.n_gpu[0],
        cache.r2_het_hist_gpu,
        cache.ld_scores_gpu,
        nbin_r2_het_hist,
    )
    return cache.sum_result()


def cost_2d_gpu(z0_1_vec, z0_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0, n_1_vec, n_2_vec,
                r2_het_hist, ld_scores, nbin_r2_het_hist):
    p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0 = map(
        nb.float32, (p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0)
    )
    cache = get_gpu_cache_2d(z0_1_vec, z0_2_vec, n_1_vec, n_2_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    base.log_pdf_2d[cache.blocks_per_grid, THREADS_PER_BLOCK](
        cache.res_gpu,
        cache.z_gpu[0],
        cache.z_gpu[1],
        p_1,
        p_2,
        sb2_1,
        sb2_2,
        s02_1,
        s02_2,
        pp,
        rho,
        rho0,
        cache.n_gpu[0],
        cache.n_gpu[1],
        cache.r2_het_hist_gpu,
        cache.ld_scores_gpu,
        nbin_r2_het_hist,
    )
    return cache.sum_result()


def cost_3d_gpu(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23,
                p_123, r2_het_hist, ld_scores, nbin_r2_het_hist):
    p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23, p_123 = map(
        nb.float64,
        (p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23, p_123),
    )
    cache = get_gpu_cache_3d(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    base.log_pdf_3d[cache.blocks_per_grid, THREADS_PER_BLOCK](
        cache.res_gpu,
        cache.z_gpu[0],
        cache.z_gpu[1],
        cache.z_gpu[2],
        cache.n_gpu[0],
        cache.n_gpu[1],
        cache.n_gpu[2],
        p_1,
        p_2,
        p_3,
        sb2_1,
        sb2_2,
        sb2_3,
        s02_1,
        s02_2,
        s02_3,
        p_12,
        p_13,
        p_23,
        rho_12,
        rho_13,
        rho_23,
        rho0_12,
        rho0_13,
        rho0_23,
        p_123,
        cache.r2_het_hist_gpu,
        cache.ld_scores_gpu,
        nbin_r2_het_hist,
    )
    return cache.sum_result()


def patch_base_module():
    base.select_snps = select_snps
    base.load_opt_data = load_opt_data
    base.get_total_het_used = get_total_het_used
    base.cost_1d_gpu = cost_1d_gpu
    base.cost_2d_gpu = cost_2d_gpu
    base.cost_3d_gpu = cost_3d_gpu


def main():
    global CPU_WORKERS_OVERRIDE
    patch_base_module()
    args = base.parse_args(base.sys.argv[1:])
    print(f"Loading config from: {args.config}")
    base.report_cuda_status()
    config = base.load_config(args.config)
    CPU_WORKERS_OVERRIDE = {
        "default": config.get("cpu_workers"),
        "pruning": config.get("cpu_workers_pruning"),
        "ld_prep": config.get("cpu_workers_ld_prep"),
        "total_het": config.get("cpu_workers_total_het"),
    }
    maxiter_2d_glob, maxiter_2d_loc = get_bivariate_optimization_iters(config)

    nbin_het_hist = config["nbin_het_hist"]
    print(f"{nbin_het_hist} bins in het hist.")

    base.log_phase("Start loading SNPs and summary statistics")
    snps_df = base.load_snps(
        config["template_dir"],
        config["sumstats"],
        chromosomes=config["snp_filters"]["chromosomes"],
        z_thresh=config["snp_filters"]["z_thresh"],
        info_thresh=config["snp_filters"]["info_thresh"],
        maf_thresh=config["snp_filters"]["maf_thresh"],
        exclude_regions=config["snp_filters"]["exclude_regions"],
    )
    base.log_phase("Finished loading SNPs and summary statistics")

    base.log_phase("Start SNP selection and pruning")
    snps2keep = base.select_snps(
        snps_df,
        snps2keep=None,
        n_random=config["pruning"]["n_random"],
        do_pruning=config["pruning"]["do_pruning"],
        r2_prune_thresh=config["pruning"]["r2_prune_thresh"],
        template_dir=config["template_dir"],
        rng_seed=config["pruning"]["rand_prune_seed"],
    )
    base.log_phase("Finished SNP selection and pruning")

    base.log_phase("Start LD data preparation")
    r2_het_hist, z_n_dict, ld_scores = base.load_opt_data(
        config["template_dir"],
        snps_df,
        snps2keep=snps2keep,
        nbin_het_hist=nbin_het_hist,
    )
    base.log_phase("Finished LD data preparation")

    r2_het_hist_global, z_n_dict_global, ld_scores_global = r2_het_hist, z_n_dict, ld_scores

    base.log_phase("Start total heterozygosity calculation")
    total_used_het = base.get_total_het_used(
        config["template_dir"],
        snps_df,
        config["pruning"]["rand_prune_seed"],
        config["pruning"]["r2_prune_thresh"],
    )
    base.log_phase("Finished total heterozygosity calculation")

    if True:
        base.log_phase("Start univariate optimization 1")
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_1 = base.optimize_1d(
            z_n_dict["Z_0"],
            z_n_dict["N_0"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_0"],
            z_n_dict_global["N_0"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
            maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"],
        )
        opt_out_1["h2"] = total_used_het * opt_out_1["opt_par"][0] * opt_out_1["opt_par"][1]
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 1:")
        print(opt_out_1)
        base.log_phase("Finished univariate optimization 1")

    if True:
        base.log_phase("Start univariate optimization 2")
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_2 = base.optimize_1d(
            z_n_dict["Z_1"],
            z_n_dict["N_1"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_1"],
            z_n_dict_global["N_1"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
            maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"],
        )
        opt_out_2["h2"] = total_used_het * opt_out_2["opt_par"][0] * opt_out_2["opt_par"][1]
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 2:")
        print(opt_out_2)
        base.log_phase("Finished univariate optimization 2")

    if True:
        base.log_phase("Start univariate optimization 3")
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_3 = base.optimize_1d(
            z_n_dict["Z_2"],
            z_n_dict["N_2"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_2"],
            z_n_dict_global["N_2"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
            maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"],
        )
        opt_out_3["h2"] = total_used_het * opt_out_3["opt_par"][0] * opt_out_3["opt_par"][1]
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 3:")
        print(opt_out_3)
        base.log_phase("Finished univariate optimization 3")

    if True:
        p_1, sb2_1, s02_1 = opt_out_1["opt_par"]
        p_2, sb2_2, s02_2 = opt_out_2["opt_par"]
        base.log_phase("Start bivariate optimization 1 vs 2")
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_12 = base.optimize_2d(
            p_1,
            sb2_1,
            s02_1,
            z_n_dict["N_0"],
            z_n_dict["Z_0"],
            p_2,
            sb2_2,
            s02_2,
            z_n_dict["N_1"],
            z_n_dict["Z_1"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_0"],
            z_n_dict_global["N_0"],
            z_n_dict_global["Z_1"],
            z_n_dict_global["N_1"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_2d_glob=maxiter_2d_glob,
            maxiter_2d_loc=maxiter_2d_loc,
        )
        opt_out_12["rg"] = opt_out_12["opt_par"][0] * opt_out_12["opt_par"][1] / math.sqrt(p_1 * p_2)
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 1 vs 2:")
        print(opt_out_12)
        base.log_phase("Finished bivariate optimization 1 vs 2")

    if True:
        p_1, sb2_1, s02_1 = opt_out_1["opt_par"]
        p_3, sb2_3, s02_3 = opt_out_3["opt_par"]
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_13 = base.optimize_2d(
            p_1,
            sb2_1,
            s02_1,
            z_n_dict["N_0"],
            z_n_dict["Z_0"],
            p_3,
            sb2_3,
            s02_3,
            z_n_dict["N_2"],
            z_n_dict["Z_2"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_0"],
            z_n_dict_global["N_0"],
            z_n_dict_global["Z_2"],
            z_n_dict_global["N_2"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_2d_glob=maxiter_2d_glob,
            maxiter_2d_loc=maxiter_2d_loc,
        )
        opt_out_13["rg"] = opt_out_13["opt_par"][0] * opt_out_13["opt_par"][1] / math.sqrt(p_1 * p_3)
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 1 vs 3:")
        print(opt_out_13)

    if True:
        p_2, sb2_2, s02_2 = opt_out_2["opt_par"]
        p_3, sb2_3, s02_3 = opt_out_3["opt_par"]
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_23 = base.optimize_2d(
            p_2,
            sb2_2,
            s02_2,
            z_n_dict["N_1"],
            z_n_dict["Z_1"],
            p_3,
            sb2_3,
            s02_3,
            z_n_dict["N_2"],
            z_n_dict["Z_2"],
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            z_n_dict_global["Z_1"],
            z_n_dict_global["N_1"],
            z_n_dict_global["Z_2"],
            z_n_dict_global["N_2"],
            r2_het_hist_global,
            ld_scores_global,
            maxiter_2d_glob=maxiter_2d_glob,
            maxiter_2d_loc=maxiter_2d_loc,
        )
        opt_out_23["rg"] = opt_out_23["opt_par"][0] * opt_out_23["opt_par"][1] / math.sqrt(p_2 * p_3)
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 2 vs 3:")
        print(opt_out_23)

    if True:
        p_1, sb2_1, s02_1 = opt_out_1["opt_par"]
        p_2, sb2_2, s02_2 = opt_out_2["opt_par"]
        p_3, sb2_3, s02_3 = opt_out_3["opt_par"]
        p_12, rho_12, rho0_12 = opt_out_12["opt_par"]
        p_13, rho_13, rho0_13 = opt_out_13["opt_par"]
        p_23, rho_23, rho0_23 = opt_out_23["opt_par"]

        p_123_lb, p_123_rb = math.log10(max(1e-6, p_12 + p_13 - p_1, p_12 + p_23 - p_2, p_13 + p_23 - p_3)), math.log10(min(p_12, p_13, p_23))
        if p_123_lb > p_123_rb:
            print("Run triple bivariate analysis to make parameters feasible for trivariate.")
            now = base.datetime.now()
            start_time = now.strftime("%D-%H:%M:%S")
            opt_out_12_13_23 = base.optimize_2d_constr(
                p_1,
                sb2_1,
                s02_1,
                p_2,
                sb2_2,
                s02_2,
                p_3,
                sb2_3,
                s02_3,
                p_12,
                rho_12,
                rho0_12,
                p_13,
                rho_13,
                rho0_13,
                p_23,
                rho_23,
                rho0_23,
                z_n_dict["Z_0"],
                z_n_dict["Z_1"],
                z_n_dict["Z_2"],
                z_n_dict["N_0"],
                z_n_dict["N_1"],
                z_n_dict["N_2"],
                r2_het_hist,
                ld_scores,
                nbin_het_hist,
                config["optimization"].get("maxiter_2d_constr", 20),
            )
            now = base.datetime.now()
            end_time = now.strftime("%D-%H:%M:%S")
            print("Start Time =", start_time)
            print("End Time =", end_time)
            print("Bivariate constrained result 1 vs 2 vs 3:")
            print(opt_out_12_13_23)
            p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23 = opt_out_12_13_23["opt_par"]
            opt_out_12_13_23["opt_par"].append(p_12 * rho_12 / math.sqrt(p_1 * p_2))
            opt_out_12_13_23["opt_par"].append(p_13 * rho_13 / math.sqrt(p_1 * p_3))
            opt_out_12_13_23["opt_par"].append(p_23 * rho_23 / math.sqrt(p_2 * p_3))
        else:
            opt_out_12_13_23 = None
        now = base.datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_123 = base.optimize_3d(
            z_n_dict["Z_0"],
            z_n_dict["Z_1"],
            z_n_dict["Z_2"],
            z_n_dict["N_0"],
            z_n_dict["N_1"],
            z_n_dict["N_2"],
            p_1,
            p_2,
            p_3,
            sb2_1,
            sb2_2,
            sb2_3,
            s02_1,
            s02_2,
            s02_3,
            p_12,
            p_13,
            p_23,
            rho_12,
            rho_13,
            rho_23,
            rho0_12,
            rho0_13,
            rho0_23,
            r2_het_hist,
            ld_scores,
            nbin_het_hist,
            config["optimization"]["maxiter_3d"],
        )
        now = base.datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Trivariate result 1 vs 2 vs 3:")
        print(opt_out_123)

    out_dict = dict(
        config=config,
        opt_out_1=opt_out_1,
        opt_out_2=opt_out_2,
        opt_out_3=opt_out_3,
        opt_out_12=opt_out_12,
        opt_out_13=opt_out_13,
        opt_out_23=opt_out_23,
        opt_out_12_13_23=opt_out_12_13_23,
        opt_out_123=opt_out_123,
    )

    with open(config["out"], "w") as f:
        base.json.dump(out_dict, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
