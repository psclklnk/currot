import argparse
from misc.util import select_best_hps, param_comp
from misc.generate_hp_search_scripts import alg_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="cluster_logs")
    parser.add_argument("--learner", type=str, default="sac", choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="maze", choices=["point_mass_2d", "maze"])

    args = parser.parse_args()
    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment
    elif args.env == "maze":
        from deep_sprl.experiments import MazeExperiment
        exp = MazeExperiment
    else:
        raise RuntimeError("Unknown environment: %s" % args.env)

    for method, params in alg_params.items():
        top_hps = select_best_hps(exp, args.learner, method, {k: v for (k, v) in params}, args.base_log_dir)
        param_comp(exp, args.learner, teacher=method, full_params=top_hps, log_dir=args.base_log_dir)


if __name__ == "__main__":
    main()
