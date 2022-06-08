import os
import argparse

alg_params = {
    "alp_gmm": [("AG_P_RAND", [0.1, 0.2, 0.3]), ("AG_FIT_RATE", [50, 100, 200]), ("AG_MAX_SIZE", [500, 1000, 2000])],
    "goal_gan": [("GG_NOISE_LEVEL", [0.025, 0.05, 0.1]), ("GG_FIT_RATE", [50, 100, 200]),
                 ("GG_P_OLD", [0.1, 0.2, 0.3])],
    "acl": [("ACL_EPS", [0.05, 0.1, 0.2]), ("ACL_ETA", [0.01, 0.025, 0.05])],
    "plr": [("PLR_REPLAY_RATE", [0.55, 0.7, 0.85]), ("PLR_BETA", [0.15, 0.3, 0.45]), ("PLR_RHO", [0.15, 0.3, 0.45])],
    "vds": [("VDS_LR", [0.0001, 0.0005, 0.001]), ("VDS_EPOCHS", [3, 5, 10]), ("VDS_BATCHES", [20, 40, 80])]
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="alp_gmm", choices=["alp_gmm", "goal_gan", "acl", "plr", "vds"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="point_mass_2d", choices=["point_mass_2d", "maze"])
    parser.add_argument("--n_seeds", default=1, type=int)
    parser.add_argument("--n_cores", default=1, type=int)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_dir, "hp_search_script_template.sh"), "r") as f:
        raw_lines = f.readlines()

    params = alg_params[args.type]
    extra_args = [""]
    for param_name, param_values in params:
        new_extra_args = []
        for cur_arg in extra_args:
            for param_value in param_values:
                new_extra_args.append(cur_arg + "--{0} {1} ".format(param_name, param_value))
        extra_args = new_extra_args

    for i, cur_extra_args in enumerate(extra_args):
        formatted_lines = []
        cur_params = vars(args)
        cur_params["extra_args"] = cur_extra_args
        for line in raw_lines:
            formatted_lines.append(line.format(**cur_params))

        script_dir = os.path.join(base_dir, "cluster_scripts", "hp_search", args.type, args.env, args.learner)
        os.makedirs(script_dir, exist_ok=True)

        with open(os.path.join(script_dir, "run_%d.sh" % i), "w") as f:
            f.writelines(formatted_lines)


if __name__ == "__main__":
    main()
