from algo.storm import STORM_LVC
from envs import make_vec_envs
from utils import *
from models import *
from storage import RolloutStorage
from collections import deque
from itertools import count
from tensorboardX import SummaryWriter


GAMMA = 0.99
ACTOR_LR = 3e-2
CRITIC_LR = 3e-3
SEED = 11
CUDA = True
NUM_INNER = 10
ENV_NAME = "HalfCheetah-v2"
OUTER_BATCHSIZE = 10000
INNER_BATCHSIZE = 10000
NUM_PROCESS = 1

torch.set_num_threads(NUM_PROCESS)
set_seed(SEED)
device = torch.device("cuda:0" if CUDA else "cpu")
logdir = "./GD_STORM_LVC/%s/batchsize%d_innersize%d_seed%d_lrcritic%f_lractorinit%f_freq_%d" % (
    str(ENV_NAME), OUTER_BATCHSIZE, INNER_BATCHSIZE, SEED, CRITIC_LR, ACTOR_LR, NUM_INNER)
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
agent = STORM_LVC(
    actor=actor,
    critic=critic,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    alpha_initial=1)

rollouts = RolloutStorage(
    num_steps=OUTER_BATCHSIZE,
    num_processes=NUM_PROCESS,
    obs_shape=envs.observation_space.shape,
    action_space=envs.action_space,
    recurrent_hidden_state_size=1)

inner_rollouts = RolloutStorage(
    num_steps=INNER_BATCHSIZE,
    num_processes=NUM_PROCESS,
    obs_shape=envs.observation_space.shape,
    action_space=envs.action_space,
    recurrent_hidden_state_size=1)


obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)
inner_rollouts.obs[0].copy_(obs)
inner_rollouts.to(device)
episode_rewards = deque(maxlen=10)
total_num_steps = 0

def select_action(obs):
    with torch.no_grad():
        action_mean, log_std = actor(obs)
        action = torch.normal(action_mean, torch.exp(log_std))
        var = torch.exp(log_std) ** 2
        action_log_probs = -((action - action_mean) ** 2) / (2 * var) - log_std - math.log(
            math.sqrt(2 * math.pi))
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        return action, action_log_probs

def sample(batchsize, ro):
    for step in range(batchsize//NUM_PROCESS):
        # Sample actions
        with torch.no_grad():
            action, action_log_prob = select_action(ro.obs[step])
            value = critic(ro.obs[step])
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
        ro.insert(obs, torch.tensor(0.0), action,
                        action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = critic(ro.obs[-1]).detach()

    # process sample
    ro.compute_returns(next_value, True, GAMMA, 0.97, True)

for j in count():
    # sample
    sample(OUTER_BATCHSIZE, rollouts)
    # compute updated params
    prev_params = get_flat_params_from(actor)
    value_loss, action_loss, grad, d_theta = agent.update(rollouts)
    cur_params = get_flat_params_from(actor)
    # reset envs
    rollouts.obs[0].copy_(envs.reset())
    total_num_steps += OUTER_BATCHSIZE

    writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
    writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)

    for i in range(NUM_INNER):
        agent.iteration = i+2
        a = np.random.uniform()
        mix_params = a * prev_params + (1 - a) * cur_params
        set_flat_params_to(actor, mix_params)
        sample(INNER_BATCHSIZE, inner_rollouts)
        set_flat_params_to(actor, cur_params)

        prev_params = cur_params
        value_loss, action_loss, grad, d_theta = agent.inner_update(rollouts, grad, d_theta)
        inner_rollouts.obs[0].copy_(envs.reset())
        cur_params = get_flat_params_from(actor)
        total_num_steps += INNER_BATCHSIZE

        writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
        writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)
        print("Inner_updates {}, num timesteps {}, mean_reward {:.1f}"
              .format(i+1, total_num_steps, np.mean(episode_rewards)))

    print(
        "Updates {}, num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        .format(j+1, total_num_steps,
                len(episode_rewards), np.mean(episode_rewards),
                np.median(episode_rewards), np.min(episode_rewards),
                np.max(episode_rewards), value_loss,
                action_loss))
    if total_num_steps > 4e6:
        break