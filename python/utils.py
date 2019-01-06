import time
from inspect import getframeinfo, currentframe
from config import *



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

_FAST_LOGGING = False
_start_time = time.time()

class TextColors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARN      = '\033[1;33m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


def _format(msg):
    info = getframeinfo(currentframe().f_back.f_back)
    base = "[%6.3f] (%s, line %4d)" % (time.time() - _start_time, os.path.basename(info.filename), info.lineno)
    l = len(base)
    if l < 40:
        base += ' '*(40 - l)
    return base + ": " + msg


def error(msg, **kwargs):
    exit = "noexit" not in kwargs
    fast = "fast" in kwargs
    kwargs.pop("noexit", None)
    kwargs.pop("fast", None)
    msg = msg if _FAST_LOGGING or fast else _format(msg)
    print((TextColors.FAIL + msg + TextColors.ENDC), **kwargs)
    sys.stdout.flush()
    if exit:
        sys.exit()


def warn(msg, **kwargs):
    fast = "fast" in kwargs
    kwargs.pop("fast", None)
    msg = msg if _FAST_LOGGING or fast else _format(msg)
    print((TextColors.WARN + msg + TextColors.ENDC), **kwargs)
    sys.stdout.flush()
    if "end" in kwargs and not kwargs["end"]:
        sys.stdout.flush()


def info(msg, **kwargs):
    fast = "fast" in kwargs
    kwargs.pop("fast", None)
    msg = msg if _FAST_LOGGING or fast else _format(msg)
    print((TextColors.OKBLUE + msg + TextColors.ENDC), **kwargs)
    sys.stdout.flush()
    if "end" in kwargs and not kwargs["end"]:
        sys.stdout.flush()