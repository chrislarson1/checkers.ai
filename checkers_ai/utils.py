from checkers_ai.config import *


def discounted_rewards(rewards):
    T = len(rewards)
    gamma_vec = np.array([GAMMA ** t for t in range(T)])
    disc_rewards = np.zeros_like(rewards)
    for i in range(T):
        gamma_sub_vec = gamma_vec[:T - i]
        reward_sub_vec = rewards[i: T]
        disc_rewards[i] = np.dot(gamma_sub_vec.T, reward_sub_vec)
    return disc_rewards


def advantage(rewards, values):
    T = len(rewards)
    gamma_vec = np.array([GAMMA ** t for t in range(T)])
    disc_rewards = np.zeros_like(rewards)
    for i in range(T):
        gamma_sub_vec = gamma_vec[:T - i]
        reward_sub_vec = rewards[i: T]
        disc_rewards[i] = np.dot(gamma_sub_vec.T, reward_sub_vec) - values[i]
    return disc_rewards


def scale_outputs(outputs):
    return (outputs - VALUE_RANGE[0]) / float(VALUE_RANGE[1] - VALUE_RANGE[0])


def print_stats(type_='', accuracy=None):
    print('{0}:'.format(type_))
    print('  top {0} probabilities: {1}'.format(10, accuracy, sum(accuracy)))
    print('  top 5 cummulative: {0}'.format(sum(accuracy[:5])))
    print('  top 10 cummulative: {0}'.format(sum(accuracy)))
