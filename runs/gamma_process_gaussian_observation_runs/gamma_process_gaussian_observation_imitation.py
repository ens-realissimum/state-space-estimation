import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import bayesian_framework.filters as b_filters
import bayesian_framework.shared.numerical_computations as num_cmp
from bayesian_framework.bayesian_filter_type import BayesianFilterType, is_linear_kalman_filter, is_sigma_point_filter, is_sqrt_sigma_point_filter
from bayesian_framework.inference.inference_model_generator import build_filterable_model
from bayesian_framework.inference.stochastic_models.covariance_type import CovarianceType
from bayesian_framework.inference.stochastic_models.noise_type import NoiseType
from bayesian_framework.inference.stochastic_models.stochastic_models import build_stochastic_process
from bayesian_framework.state_space_models import gssm_gamma_proc_gauss_observ as gamma_gssm
from utils.stat_utils import stat_errors


def run():
    # todo: check dim of return between different numeric methods
    # gh_points, gh_weights = num_cmp.eval_gauss_hermite_rule(4, 3)
    # todo: fix generate_sparse_gauss_hermite_points
    # sgh_points, sgh_weights = num_cmp.generate_sparse_gauss_hermite_points(2, 3, 2)

    filter_types = [BayesianFilterType.sghqf]  # todo: fix sghqf
    # [kf, ekf, ukf, srukf, cdkf, srcdkf, ckf, srckf, fdckf, cqkf, ghqf, sghqf, pf, gspf ??,  ]

    number_of_runs = 100  # 500
    data_points_count = 100  # kf and ekf works only when <= 30
    draw_iterations = True
    err_arr = np.zeros((number_of_runs, data_points_count - 1))

    for filter_type in filter_types:
        print(filter_type.name)

        fig_a, ax_a = plt.subplots()

        gssm_model = gamma_gssm.build(4e-2, 0.5, 3, 0.5, 0, 1e-2, CovarianceType.sqrt)
        inference_model = build_filterable_model(gssm_model, filter_type)

        for i in range(number_of_runs):
            x = np.zeros((gssm_model.state_dim, data_points_count), dtype=np.float64)
            z = np.zeros((gssm_model.observation_dim, data_points_count), dtype=np.float64)

            x_noise = gssm_model.state_noise.sample(data_points_count)
            z_noise = gssm_model.observation_noise.sample(data_points_count)

            z[0] = gssm_model.observation_func(x[:, 0], z_noise[:, 0], np.asarray([1]))
            for j in range(1, data_points_count):
                x[:, j] = gssm_model.transition_func(x[:, j - 1], x_noise[:, j - 1], np.asarray([j - 1]))
                z[:, j] = gssm_model.observation_func(x[:, j], z_noise[:, j], np.asarray([j]))

            u1 = np.asarray(range(data_points_count))
            u2 = np.asarray(range(1, data_points_count + 1))

            x_est = np.zeros((inference_model.state_dim, data_points_count))
            x_est[:, 0] = 1
            x_cov_est = 3 / 4 * np.eye(inference_model.state_dim)

            if is_sigma_point_filter(filter_type) or is_linear_kalman_filter(filter_type):
                if filter_type is BayesianFilterType.ukf:
                    spkf = b_filters.Ukf(alpha=1, beta=2, kappa=0)
                elif filter_type is BayesianFilterType.srukf:
                    spkf = b_filters.SrUkf(alpha=1, beta=2, kappa=0)
                elif filter_type is BayesianFilterType.cdkf:
                    spkf = b_filters.Cdkf(scale_factor=np.sqrt(3))
                elif filter_type is BayesianFilterType.srcdkf:
                    spkf = b_filters.SrCdkf(scale_factor=np.sqrt(3))
                elif filter_type is BayesianFilterType.ckf:
                    spkf = b_filters.Ckf()
                elif filter_type is BayesianFilterType.srckf:
                    spkf = b_filters.SrCkf()
                elif filter_type is BayesianFilterType.fdckf:
                    spkf = b_filters.FdCkf()
                elif filter_type is BayesianFilterType.cqkf:
                    spkf = b_filters.Cqkf(order=9)
                elif filter_type is BayesianFilterType.ghqf:
                    spkf = b_filters.Ghqf(order=11)
                elif filter_type is BayesianFilterType.sghqf:
                    spkf = b_filters.Sghqf(order=11, manner=3)
                elif filter_type is BayesianFilterType.kf:
                    spkf = b_filters.Kf()
                elif filter_type is BayesianFilterType.ekf:
                    spkf = b_filters.Ekf()
                else:
                    raise Exception("Not supported filter type: {0}".format(filter_type.name))

                if is_sqrt_sigma_point_filter(filter_type):
                    x_cov_est = np.linalg.cholesky(x_cov_est)

                for k in range(1, data_points_count):
                    x_est[:, k], x_cov_est, _ = spkf.estimate(x_est[:, k - 1], x_cov_est, z[:, k], inference_model, u1[k - 1], u2[k])

            elif filter_type is BayesianFilterType.pf:
                resample_strategy = b_filters.ResampleStrategy.resolve(b_filters.ResampleType.residual)
                pf = b_filters.Pf(0.1, resample_strategy=resample_strategy)
                n_particles = int(1e5)
                particles = np.atleast_2d(multivariate_normal.rvs(x_est[:, 0], x_cov_est, n_particles))
                weights = np.tile(1 / n_particles, n_particles)
                data_set = b_filters.BootstrapDataSet(particles, weights)

                for k in range(1, data_points_count):
                    x_est[:, k], data_set = pf.estimate(data_set, z[:, k], inference_model, u1[k - 1], u2[k])
            elif filter_type is BayesianFilterType.gspf:
                n_particles = int(1e3)
                n_mixture = 4
                initial_cov = np.squeeze(inference_model.state_noise.covariance)
                gm_cov = np.zeros((4, 1, 1))
                gm_cov[0, :, :] = 2 * initial_cov
                gm_cov[1, :, :] = 0.5 * initial_cov
                gm_cov[2, :, :] = 1.5 * initial_cov
                gm_cov[3, :, :] = 1.25 * initial_cov
                gm_state_noise = build_stochastic_process(
                    NoiseType.gaussian_mixture,
                    mixture_size=n_mixture,
                    mean=np.tile(np.asarray(np.squeeze(inference_model.state_noise.mean)), (n_mixture, 1)),
                    covariance=gm_cov,
                    covariance_type=inference_model.state_noise.covariance_type,
                    weights=np.asarray([0.45, 0.45, 0.05, 0.05])
                )
                gmm_inference_model = inference_model.replace_state_noise(gm_state_noise)

                resample_strategy = b_filters.ResampleStrategy.resolve(b_filters.ResampleType.residual)
                gspf = b_filters.Gspf(0.001, n_particles, resample_strategy=resample_strategy)
                gmm = b_filters.init_gmm(x_est[:, 0], x_cov_est, n_particles, n_mixture)
                for k in range(1, data_points_count):
                    x_est[:, k], data_set = gspf.estimate(gmm, z[:, k], gmm_inference_model, u1[k - 1], u2[k])
            else:
                raise Exception("Not supported filter type: {0}".format(filter_type.name))

            if draw_iterations and i == 1:
                ax_a.plot(np.squeeze(x), linewidth=2.0, label="clean", linestyle="dashed")
                ax_a.plot(np.squeeze(z), linewidth=2.0, label="noisy", linestyle="dotted")
                ax_a.plot(np.squeeze(x_est), linewidth=2.0, label="{0} estimate".format(filter_type.name))
                ax_a.grid(True, which="both", axis="both")
                ax_a.legend(loc="upper right")
                ax_a.set_title("{0}: Nonlinear Time Variant State Estimation \n (non Gaussian noise)".format(filter_type.name), fontsize=12)
                plt.tight_layout()
                fig_a.set_size_inches(8, 6)
                fig_a.show()

            err_arr[i, :] = x_est[:, 1:] - x[:, 1:]

            if ((i + 1) * 100 / number_of_runs) % 10 == 0.0:
                print("\t{0} % completed".format((i + 1) * 100 / number_of_runs))

        x_mean_err, std_x_err, rmse_x_err = stat_errors(err_arr)

        fig1, ax1 = plt.subplots()
        ax1.plot(x_mean_err, color="b", linewidth=2.0, label="mean rmse")
        ax1.plot(x_mean_err - std_x_err, color="r", linewidth=2.0, label="-1 sigma", linestyle="dashed")
        ax1.plot(x_mean_err + std_x_err, color="r", linewidth=2.0, label="+1 sigma", linestyle="dashed")
        ax1.legend(loc="upper right")
        ax1.grid(True, which="both", axis="both")
        plt.tight_layout()
        fig1.show()

        fig2, ax2 = plt.subplots()
        ax2.plot(rmse_x_err, color="g", linewidth=2.0)
        ax2.set_title("{0}: RMSE (non Gaussian noise)".format(filter_type.name), fontsize=12)
        ax2.grid(True, which="both", axis="both")
        fig2.show()
