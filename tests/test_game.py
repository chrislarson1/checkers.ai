import os
import time
from tqdm import tqdm
import numpy as np
from checkers_ai.config import POLICY_PATH
from checkers_ai.game import Game
from checkers_ai.player import PolicyPlayer


def test_game_policy_policy():

    # np.random.seed(1234)

    t0 = time.time()

    player1 = PolicyPlayer(
        load_dir=os.path.join(POLICY_PATH, '81.9acc'),
        selection='eps-greedy',
        device='GPU',
        gpu_idx=None
    )

    player2 = PolicyPlayer(
        load_dir=os.path.join(POLICY_PATH, '72acc'),
        selection='eps-greedy',
        device='GPU',
        gpu_idx=None
    )

    # Simulation
    checkers = Game(player1=player1,
                    player2=player2,
                    verbose=False)

    N = 50
    bar = tqdm(total=N)
    score = [0, 0, 0]
    for i in range(N):
        checkers.play(first_move=np.random.choice([-1, 1]))
        if checkers.winner == 1:
            score[0] += 1
        elif checkers.winner == -1:
            score[1] += 1
        else:
            score[2] += 1
        bar.update(1)
        bar.set_description("Running simulations: ")
    bar.close()
    t = time.time() - t0
    print("Total time: {}".format(t))
    print("Average game time: {} sec".format(t / N))
    print("aborted_games: {}".format(checkers.aborted_games))
    print("Policy1 wins: {0} | Policy2 wins: {1} | Draws: {2}".format(*score))

    # print(checkers.traj[1].discounted_rewards(step=15))
    # print(checkers.traj[-1].discounted_rewards(step=15))


if __name__ == '__main__':
    test_game_policy_policy()
