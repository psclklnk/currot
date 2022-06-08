python run.py --type self_paced --learner ppo --env point_mass_2d --seed $1
python run.py --type wasserstein --learner ppo --env point_mass_2d --seed $1

python run.py --type wasserstein --learner sac --env maze --seed $1
python run.py --type self_paced --learner sac --env maze --seed $1
