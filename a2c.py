from algo.a2c import A2C
from envs import make_vec_envs
from utils import *
from models import *
from storage import RolloutStorage
from collections import deque
import time
from tensorboardX import SummaryWriter

GAMMA = 0.99
ACTOR_LR = 3e-3
CRITIC_LR = 3e-3
SEED = 1
CUDA = True
NUM_ITER = 100
ENV_NAME = "Hopper-v2"
BATCHSIZE = 10000
NUM_PROCESS = 10

torch.set_num_threads(NUM_PROCESS)
set_seed(SEED)
device = torch.device("cuda:0" if CUDA else "cpu")
logdir = "./GD_SGD_LVC/%s/batchsize%d_seed%d_lrcritic%f_lractorinit%f" % (
    str(ENV_NAME), BATCHSIZE, SEED, CRITIC_LR, ACTOR_LR)
writer = SummaryWriter(log_dir=logdir)

envs = make_vec_envs(
    env_name=ENV_NAME,
    seed=SEED,
    num_processes=NUM_PROCESS,
    gamma=GAMMA,
    log_dir='./env_log/',
    device=device,
    allow_early_resets=True)
actor = Policy(
    num_inputs=envs.observation_space.shape[0],
    num_outputs=envs.action_space.shape[0],
    hidden_size=64)
critic = Value(
    num_inputs=envs.observation_space.shape[0],
    hidden_size=64)
actor.to(device)
critic.to(device)
agent = A2C(
    actor=actor,
    critic=critic,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR)

rollouts = RolloutStorage(
    num_steps=BATCHSIZE,
    num_processes=NUM_PROCESS,
    obs_shape=envs.observation_space.shape,
    action_space=envs.action_space,
    recurrent_hidden_state_size=1)

obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)
episode_rewards = deque(maxlen=10)
start = time.time()

def select_action(obs):
    with torch.no_grad():
        action_mean, log_std = actor(obs)
        action = torch.normal(action_mean, torch.exp(log_std))
        var = torch.exp(log_std) ** 2
        action_log_probs = -((action - action_mean) ** 2) / (2 * var) - log_std - math.log(
            math.sqrt(2 * math.pi))
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        return action, action_log_probs

def sample(batchsize, rollouts):
    for step in range(BATCHSIZE//NUM_PROCESS):
        # Sample actions
        with torch.no_grad():
            action, action_log_prob = select_action(rollouts.obs[step])
            value = critic(rollouts.obs[step])
        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, torch.tensor(0.0), action,
                        action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = critic(rollouts.obs[-1]).detach()

    # process sample
    rollouts.compute_returns(next_value, True, GAMMA, 0.97, True)

for j in range(NUM_ITER):
    # sample
    sample(BATCHSIZE, rollouts)
    # compute updated params
    value_loss, action_loss = agent.update(rollouts)
    # reset envs
    rollouts.obs[0].copy_(envs.reset())

    total_num_steps = (j + 1) * BATCHSIZE
    writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
    end = time.time()
    print(
        "Updates {}, num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        .format(j, total_num_steps,
                len(episode_rewards), np.mean(episode_rewards),
                np.median(episode_rewards), np.min(episode_rewards),
                np.max(episode_rewards), value_loss,
                action_loss))