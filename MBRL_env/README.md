# MBRL_env

This repository holds a collection of environments for MBRL.

## Env List

Should be working

1. 'MBRLHalfCheetah-v0'
2. 'MBRLReacher3D-v0'
3. 'gym_pendulum', 
4. 'gym_acrobot', 
5. 'gym_invertedPendulum', 
6. 'gym_ant' 

Might be working

7. 'gym_cartpole', 
8. 'gym_hopper', 
9. 'gym_reacher', 
10. 'gym_walker2d', 
11. 'gym_cheetah'

Not working yet

12. 'gym_fswimmer',
13. 'MBRLPusher-v0'

## Installation

1. Clone the repo, and copy all files to the root directory of your code base.

2. Install pip requirements by
```
cd <ROOT_DIR>
pip -r requirements.txt
```

3. Then run the following to verify that it's working (if no errors)
```
python example.py
```

## Key Dependencies (please use the exact version)

1. gym==0.17.3
2. mujoco-py==1.50.1.68

## Usage

There are two types of environments that need to be instantiated in different ways.

Type I:
```
import gym
import dmbrl.env
task = ['MBRLHalfCheetah-v0', 'MBRLReacher3D-v0', 'MBRLPusher-v0']
idx = 0
env = gym.make(task[idx])
```

Type II:
```
from mbbl.env import env_register
task = ['gym_pendulum', 'gym_acrobot', 'gym_invertedPendulum', 'gym_cartpole', 'gym_ant', 'gym_fswimmer','gym_hopper', 'gym_reacher', 'gym_walker2d', 'gym_cheetah']
# env is the environment object compatible with Gym, env_info is a dictionary with env info
idx = 0
env, env_info = env_register.make_env(task_name=task[idx], rand_seed=1234, misc_info='reset_type':'gym')
```

## Reproducibility

Random Seed are set for the following three components

### Training Seed

For the 5 runs, use training seed {1,2,3,4,5}
Add the following code at the beginning of your code to set the random seed.

```
# Comment out tf / torch based on your need
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    print("Using seed {}".format(seed))

training_seed = 1
set_seed(training_seed)
```

### Environment Seed 

```
# suppose the training env is the variable "env"
env.seed(1234)
```

### Evaluation Seed

In the beginning of a training run, initialize the evaluation environment by
```
# suppose the training env is the variable "env"
env_eval = deepcopy(env)
env_eval.seed(0)
```

## Evaluation Protocol

1. We report the mean and standard error of 5 runs.
2. During each run, the evaluation is performed by taking the average return of 5 episodes of evaluation environment, at an interval of 1k training step (200 steps for pendulum).

**Output format**:

```
# pseudo code
from scipy.io import savemat
rets, rets_ste = [], []
# test_rets is a list of 5 evaluation returns 
while in training loop:
    rets.append(np.mean(test_rets))
    rets_ste.append(np.std(test_rets)/np.sqrt(5))
    num_steps.append(steps_in_this_episode) # [200, 200, 200, 150, ...]
savemat(
    "logs.mat",
    {
        "test_returns": rets,
        "test_returns_std_err": rets_ste,
        'num_steps': num_steps
    }
)
```

To read the mat file:
```
from scipy.io import loadmat
mat_data = loadmat("logs.mat")
test_returns = mat_data['test_returns']
test_returns_std_err = mat_data['test_returns_std_err']
num_steps = mat_data['num_steps']
```