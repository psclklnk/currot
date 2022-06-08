This folder contains the code for running and visualizing the sparse goal reaching and point mass experiments presented in the main paper.
For the bipedal walker experiment, we provide the wrapper that we used to interface our CRL method to the benchmark. The interface is located at 
````
deep_sprl/teachers/spl/wasserstein_teacher_tma.py
````
and needs to be integrated into this repository: https://github.com/flowersteam/TeachMyAgent.  

For the sparse goal reaching and point mass environment, we used Python 3.8 to run the experiments. The required dependencies are listed in the **requirements.txt**
file and can be installed via
```shell script
pip install -r requirements.txt
```
The experiments can be run via the **run.py** scripts. For convenience, we have created the 
**baseline_experiments.sh** and **interpolation_based_experiments.sh** scripts for running the baselines and interpolation-based algorithms.  To execute all experiments for particular seed (in the example
below seed 1), you can run
```shell script
./baseline_experiments.sh 1
./interpolation_based_experiments.sh 1
```
After running the desired amount of seeds, you can visualize the results via
```shell script
cd misc
./visualize_results.sh
```
