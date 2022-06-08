import os
import argparse

algs = ["default", "random", "wasserstein", "alp_gmm", "goal_gan", "acl", "plr", "vds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="point_mass_2d", choices=["point_mass_2d", "maze"])
    parser.add_argument("--n_seeds", default=1, type=int)
    parser.add_argument("--n_cores", default=1, type=int)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_dir, "eval_script_template.sh"), "r") as f:
        raw_lines = f.readlines()

    for i, alg in enumerate(algs):
        formatted_lines = []
        cur_params = vars(args)
        cur_params["type"] = alg
        for line in raw_lines:
            formatted_lines.append(line.format(**cur_params))

        script_dir = os.path.join(base_dir, "cluster_scripts", "eval", args.env)
        os.makedirs(script_dir, exist_ok=True)

        with open(os.path.join(script_dir, "%s.sh" % args.type), "w") as f:
            f.writelines(formatted_lines)


if __name__ == "__main__":
    main()
