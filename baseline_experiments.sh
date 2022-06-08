python run.py --type acl --learner ppo --env point_mass_2d --seed $1
python run.py --type plr --learner ppo --env point_mass_2d --seed $1
python run.py --type vds --learner ppo --env point_mass_2d --seed $1
python run.py --type alp_gmm --learner ppo --env point_mass_2d --seed $1
python run.py --type goal_gan --learner ppo --env point_mass_2d --seed $1
python run.py --type default --learner ppo --env point_mass_2d --seed $1
python run.py --type random --learner ppo --env point_mass_2d --seed $1

python run.py --type acl --learner sac --env maze --seed $1
python run.py --type plr --learner sac --env maze --seed $1
python run.py --type vds --learner sac --env maze --seed $1
python run.py --type alp_gmm --learner sac --env maze --seed $1
python run.py --type goal_gan --learner sac --env maze --seed $1
python run.py --type random --learner sac --env maze --seed $1
python run.py --type default --learner sac --env maze --seed $1
