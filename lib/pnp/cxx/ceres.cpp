
#include <cmath>
#include <cstdio>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <omp.h>

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).

struct ReprojectionError
{
    template <typename Tf>
    ReprojectionError(Tf *pt2d, Tf *pt3d, Tf *icov_sqrtL, double *cam_K)
    {
        for (int i = 0; i < 3; ++i)
            this->pt3d[i] = pt3d[i];
        this->cam = cam_K;
        this->u = pt2d[0] - cam_K[2];
        this->v = pt2d[1] - cam_K[5];
        this->a = icov_sqrtL[0];
        this->b = icov_sqrtL[2];
        this->c = icov_sqrtL[3];
    }

    template <typename T>
    bool operator()(const T *const rt,
                    T *residuals) const
    {
        // rt[0,1,2] are the angle-axis rotation.
        T p[3];
        T srcp[3] = {T(pt3d[0]), T(pt3d[1]), T(pt3d[2])};
        ceres::AngleAxisRotatePoint(rt, srcp, p);

        // rt[3,4,5] are the translation.
        p[0] += rt[3];
        p[1] += rt[4];
        p[2] += rt[5];

        // T up = (p[0] / p[2])*fu;
        // T vp = (p[1] / p[2])*fv;
        T up = (p[0] * cam[0] + p[1] * cam[1]) / p[2];
        T vp = (p[0] * cam[3] + p[1] * cam[4]) / p[2];
        // The error is the difference between the predicted and observed position.
        T du = up - u;
        T dv = vp - v;
        residuals[0] = du * a + dv * b;
        residuals[1] = dv * c;

        return true;
    }
    template <typename Tf>
    static ceres::CostFunction *Create(Tf *pt2d, Tf *pt3d, Tf *icov_sqrtL, double *cam)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(new ReprojectionError(pt2d, pt3d, icov_sqrtL, cam));
    }
    double u, v;
    double a, b, c;
    double pt3d[3];
    double *cam;
};

#ifdef __cplusplus
extern "C"
{
#endif

    void pnp_ceres_f32(
        float *io_state_quat,
        const float *cam_K,
        const float *pts2d,
        const float *pts3d,
        const float *icov_sqrtL,
        int ptCnt,
        int maxIterCnt,
        float function_tolerance,
        int printSummary,
        float *result_tr, int *ret)
    {
        if (ptCnt < 3)
        {
            *ret = 1;
            *result_tr = 1;
            if (printSummary)
                std::cout << "skipped problem with less than 3 points\n";
            return;
        }
        double state[6];
        using namespace ceres;
        using namespace std;
        double quat[4] = {io_state_quat[0], io_state_quat[1], io_state_quat[2], io_state_quat[3]};
        QuaternionToAngleAxis(quat, state);
        for (int i = 0; i < 3; ++i)
            state[3 + i] = io_state_quat[4 + i];
        double cam[6];
        for (int i = 0; i < 6; ++i)
            cam[i] = cam_K[i];
        // Create residuals for each observation in the bundle adjustment problem. The
        // parameters for cameras and points are added automatically.
        ceres::Problem problem;
        ceres::LossFunction *loss_func = nullptr; /* default squared loss */

        for (int i = 0; i < ptCnt; ++i)
        {
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.

            ceres::CostFunction *cost_function = ReprojectionError::Create(
                &pts2d[2 * i + 0], &pts3d[3 * i], &icov_sqrtL[4 * i], cam);
            problem.AddResidualBlock(cost_function,
                                     loss_func,
                                     state);
        }

        // Make Ceres automatically detect the bundle structure. Note that the
        // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
        // for standard bundle adjustment problems.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = maxIterCnt;
        options.minimizer_progress_to_stdout = printSummary;
        options.function_tolerance = function_tolerance;
        options.logging_type = printSummary ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        AngleAxisToQuaternion(state, quat);
        if (printSummary)
            std::cout << summary.FullReport() << "\n";
        int invalid = int(!(summary.IsSolutionUsable() && summary.termination_type == ceres::CONVERGENCE));
        *ret = invalid;
        *result_tr = summary.iterations.back().trust_region_radius;
        if (invalid)
            return;
        for (int i = 0; i < 4; ++i)
            io_state_quat[i] = quat[i];
        for (int i = 0; i < 3; ++i)
        {
            io_state_quat[4 + i] = state[3 + i];
        }
    }

    void pnp_ceres_f32_omp(
        float **init_states,
        float **cam_Ks,
        float **pts2ds,
        float **pts3ds,
        float **icov_sqrtLs,
        int *ptCnts,
        int maxIterCnt,
        float function_tolerance,
        int printSummary,
        float *result_trs, int *rets,
        int job_count,
        int num_threads)
    {
        if (num_threads > 1)
        {
            omp_set_num_threads(num_threads);
#pragma omp parallel for
            for (int i = 0; i < job_count; ++i)
            {
                pnp_ceres_f32(init_states[i], cam_Ks[i], pts2ds[i], pts3ds[i], icov_sqrtLs[i], ptCnts[i], maxIterCnt, function_tolerance, printSummary, result_trs + i, rets + i);
            }
        }
        else
        {
            for (int i = 0; i < job_count; ++i)
            {
                pnp_ceres_f32(init_states[i], cam_Ks[i], pts2ds[i], pts3ds[i], icov_sqrtLs[i], ptCnts[i], maxIterCnt, function_tolerance, printSummary, result_trs + i, rets + i);
            }
        }
    }

#ifdef __cplusplus
}
#endif
