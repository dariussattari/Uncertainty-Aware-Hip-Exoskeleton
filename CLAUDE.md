# Project rules: Uncertainty-Aware Hip Exoskeleton

Layout and dataset split are described in `README.md` (Layout section).

## Cluster (FASRC Cannon) resource requests

**Why these rules exist.** In October 2026 Job Defense Shield flagged 11 of our GPU jobs: each requested 8 CPU cores
and used one (CPU utilization about 12%, which is 100% / 8 cores), wasting about 59 CPU-hours. Our training runs as a
single process: DataLoaders use `num_workers = 0` and the arrays sit in memory, so one core feeds the GPU and the
other seven idle. Over-asking also lengthens our queue times.

1. **Default every job to one core**, GPU jobs included. The GPU does the math; one core feeds it.

   ```
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=1
   ```

2. **Ask for more cores only when the code runs that many at once**, and name the reason in a comment beside the
   directive:
   - DataLoader `num_workers = N`: request N + 1;
   - `ProcessPoolExecutor` or `multiprocessing` with N workers: request N;
   - sklearn or joblib `n_jobs = N`: request N.

   The code reads the count from `$SLURM_CPUS_PER_TASK`; it never hard-codes it.
3. **Threads follow the allocation.** `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are set from `$SLURM_CPUS_PER_TASK`,
   with a fallback of 1 when it is unset.
4. **Measure before scaling out.** Run one job or one array task of any new or changed script, then run
   `jobstats <jobid>` on the cluster:
   - CPU utilization should be 90% or more of the allocated cores. If it isn't, cut `--cpus-per-task`.
   - Memory used should be close to `--mem`. Set `--mem` to the observed peak plus about 25%.
   - GPU utilization should be high. A low value means the job is waiting on data or the model is too small for a
     whole GPU.

   Launch the full array only after these numbers look right. An array multiplies a bad request by its size.
5. **Old scripts are not templates.** The scripts under `ml/sci_rob/*/slurm/` predate these rules: they request 8
   cores (4 in `pelvis_build.sbatch`), and the `env.sh` thread fallbacks are 8 and 4. Bring a script in line with
   rules 1–3 before reusing or copying it.
6. **Profile one training step before choosing a job layout.** Packing several trainings onto one GPU only helps when
   each one is CPU-bound. The Nature autoencoder was not: in fp32, cuDNN ran its convolutions as FFTs (551 ms per step
   on an A100, 98% of GPU time), and six packed copies just queued. bf16 autocast cut the step to 23 ms. Time a few
   steps and look at the top kernels (`torch.profiler`) on the target GPU type first; a MIG slice can pick different
   kernels from a full A100.
