import time
from checkers_ai.trajectory import Trajectory


def test_trajectory():

    R = Trajectory()

    for i in range(50):
        R.update(state=1, softmax=1)
    R.update(terminal_rwd=1.0)

    t0 = time.time()
    rwd = R.discounted_rewards(step=12)
    t = time.time() - t0

    print(rwd, t)


if __name__ == '__main__':
    test_trajectory()
