
from RLMultilayer.algos.ppo.ppo import ppo
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.tasks import get_env_fn
from RLMultilayer.utils import cal_reward
import os
import os.path as osp
import time
from mpi4py import MPI
import subprocess, sys
import torch

def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--env', type=str, default='PerfectAbsorberVisNIR-v0')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--discrete_thick', action="store_true")
    parser.add_argument('--maxlen', default=5, type=int)
    parser.add_argument('--hierarchical', action='store_true', help='if set to true, then output out the material type first, then condition the material thickness on the material type')
    parser.add_argument('--use_rnn', action='store_true')
    parser.add_argument('--spectrum_repr', action='store_true')
    args = parser.parse_args()

    env_kwargs = {"discrete_thick":args.discrete_thick, 'spectrum_repr':args.spectrum_repr, 'bottom_up':False, 'merit_func':cal_reward, 'maxlen':args.maxlen}


    env_fn = get_env_fn(args.env, **env_kwargs)
    epochs = 3000
    steps_per_epoch = 1000
    ac_kwargs = {"hidden_sizes":(64, 64), "cell_size":64, "not_repeat":True, "ortho_init":'on', "hierarchical":True, "channels":16, "act_emb":True, "act_emb_dim":5, "scalar_thick":False}
    use_rnn = args.use_rnn
    gamma = 1
    beta = 0.01
    lam = 0.95
    max_ep_len = args.maxlen
    actor_critic = core.RNNActorCritic if args.use_rnn else core.MLPActorCritic
    train_pi_iters = 5
    pi_lr = 5e-5
    reward_factor = 1
    spectrum_repr = args.spectrum_repr

    data_dir='./Experiments/{}'.format(args.exp_name)
    relpath = time.strftime("%Y-%m-%d") 
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=args.exp_name)
    mpi_fork(args.cpu)

    ppo(env_fn, actor_critic=actor_critic, ac_kwargs=ac_kwargs, seed=42,
        steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, beta=beta, clip_ratio=0.2, pi_lr=pi_lr,
        vf_lr=3e-4, train_pi_iters=train_pi_iters, train_v_iters=80, lam=lam, max_ep_len=max_ep_len,
        target_kl=0.01, logger_kwargs=logger_kwargs, save_freq=10, use_rnn=use_rnn, reward_factor=reward_factor, spectrum_repr=spectrum_repr)



