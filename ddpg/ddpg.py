""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

@author: Dhaval-PC

this GitHub repo has the base implementation on which I have done few changes:
https://github.com/pemami4911/deep-rl/tree/master/ddpg

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import os
from datetime import datetime
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from util import OrnsteinUhlenbeckActionNoise, Logger
# ===========================
#   Actor and Critic DNNs
# ===========================
# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars



def run_episode(env, actor, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    while not done:
        if animate:
            env.render()
        #obs = obs.astype(np.float32).reshape((1, -1))
        #obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        observes.append(obs)
        action = actor.evaluate_target(np.reshape(obs, (1, actor.s_dim))).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))
    


def run_policy(env, actor, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, actor)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    
    return np.mean([t['rewards'].sum() for t in trajectories])


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise, logger, monitor_env):
    

    # Set up summary Ops
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    env._max_episode_steps = int(args['max_episode_len'])

    for i in range(int(args['max_episodes'])):
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

#        for j in range(env.spec.max_episode_steps or int(args['max_episode_len'])):
        #terminal = False
        while True:
        #for j in range(int(args['max_episode_len'])):
            
            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                for _ in range(int(args["no_of_updates"])):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(int(args['minibatch_size']))
    
                    # Calculate targets
                    target_q = critic.predict_target(
                        s2_batch, actor.predict_target(s2_batch))
    
                    y_i = []
                    for k in range(int(args['minibatch_size'])):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + args['lambda'] * critic.gamma * target_q[k])
    
                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(
                        s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))
    
                    ep_ave_max_q += np.amax(predicted_q_value)
    
                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])
    
                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                
                if i % 20 == 0:
                    mean_reward = run_policy(monitor_env, actor, logger, 10)
                    logger.log({'Target_Reward': int(mean_reward),
                                'Total_Episode': i
                                                })
                    logger.write(display=True)
                #print('| Reward: {:d} | Episode: {:d} |'.format(int(ep_reward), \
                #        i))
                break

def main(args):
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    monitor_dir = os.path.join('videos', args['env'], "no-of-update_"+args["no_of_updates"], "random_seed"+str(args["random_seed"]))
    logger = Logger(logname=args['env'], args=args, now=now)
    with tf.Session() as sess:
        env = gym.make(args['env'])
        monitor_env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        monitor_env.seed(int(args['random_seed']))
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        
        print("****** state dimension", state_dim)
        print("****** actions dimension", action_dim)
        print("****** actions high bound", action_bound)
        
        # Ensure action bound is symmetric
        assert (np.array_equal(env.action_space.high, -env.action_space.low))

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            monitor_env = wrappers.Monitor(monitor_env, monitor_dir, force=True)

        train(sess, env, args, actor, critic, actor_noise, logger, monitor_env)
        logger.close()
        if args['use_gym_monitor']:
            env.monitor.close()
            monitor_env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--lambda', help='discount factor for critic updates', default=1)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=256)
    parser.add_argument('--no_of_updates', help='no of inner updates', default=1)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)
    main(args)
