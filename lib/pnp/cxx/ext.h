
void pnp_ceres_f32_omp(
    float ** init_states,
    float ** cam_Ks,
    float ** pts2ds,
    float ** pts3ds,
    float ** icov_sqrtLs,
    int * ptCnts,
    int maxIterCnt,
    float function_tolerance,
    int printSummary,
    float* result_trs, int* rets,
    int job_count,
    int num_threads
);


