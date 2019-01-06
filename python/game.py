import time
from tqdm import tqdm
from model import Policy
from trajectory import Trajectory
from state import State
from mcts import MCTS
from config import *


class Game:

    """
         PLAYER2
      00  01  02  03
    04  05  06  07
      08  09  10  11
    12  13  14  15
      16  17  18  19
    20  21  22  23
      24  25  26  27
    28  29  30  31
        PLAYER1
    """

    def __init__(self, player1, player2, critic1=None, critic2=None, verbose=False):
        self.state = State()
        self.player1 = player1
        self.critic1 = critic1
        self.player2 = player2
        self.critic2 = critic2
        self.verbose = verbose
        self.aborted_games = 0
        self.reset()

    def reset(self):
        self.state.init()
        self.winner = None
        self.traj = {
            1: Trajectory(),
            -1: Trajectory()
        }

    def print(self, contents):
        if self.verbose:
            if not type(contents) in (list, tuple, range):
                print(contents)
            else:
                print(contents[:5])

    def generate_action(self, player):

        contestant = self.player1 if player == 1 else self.player2
        actions_list, probs_list, actions, probs = [], [], [], []

        # ----- Human player ----- #
        if contestant == 'human':
            move_invalid = True
            while move_invalid:
                move = input("Enter move as 'pos_init pos_final':")
                if move.lower() == 'forfeit':
                    self.winner = -player
                elif move == 'draw':
                    self.winner = 'draw'
                else:
                    positions = tuple([int(x) for x in move.split(' ')])
                    if len(positions) != 2 or not all([x in range(32) for x in positions]):
                        self.print("Invalid move. Try again")
                        continue
                    actions_list.append(positions)
                    probs_list.append(1.0)
                    move_invalid = False

        # ----- Policy player ----- #
        elif isinstance(contestant, Policy):
            board_state = self.state.get_state_view(view=player)
            feed_dict = {
                contestant.state: board_state.reshape(1, 8, 4).astype(np.int32),
            }
            probs, actions = contestant.session.run(fetches=[contestant.probs, contestant.actions],
                                                    feed_dict=feed_dict)
            probs, actions = probs[0, :SELECT_N], actions[0, :SELECT_N]
            order = np.arange(0, SELECT_N)
            if contestant.selection == 'eps-greedy':
                if np.random.random() <= SELECT_EPS:
                    order = np.random.choice(order,
                                             size=SELECT_N,
                                             replace=False,
                                             p=(probs + 1e-6) / np.sum(probs + 1e-6))
            elif contestant.selection == 'multinomial':
                order = np.random.choice(order,
                                         size=SELECT_N,
                                         replace=False,
                                         p=(probs + 1e-6) / np.sum(probs + 1e-6))
            for i, (prob, action) in enumerate(zip(probs[order], actions[order])):
                move = np.zeros((128,))
                action = action if player == 1 else self.state.transform_action(action)
                move[action] = 1
                pos_init, pos_final, move_type = self.state.action2positions(move, player=player)
                if pos_init and pos_final:
                    actions_list.append((pos_init, pos_final))
                    probs_list.append(prob)

        # ----- MCTS player ----- #
        elif isinstance(contestant, MCTS):
            # Todo: Implement
            pass

        return actions_list, probs_list, actions, probs

    def status_check(self, move_count):
        n_p1 = len(np.argwhere(self.state.get_state_view() > EMPTY))
        n_p2 = len(np.argwhere(self.state.get_state_view() < EMPTY))
        piece_differential = n_p1 - n_p2
        if n_p1 == 0:
            self.winner = -1
        elif n_p2 == 0:
            self.winner = 1
        elif move_count >= 40 and \
                piece_differential > 2:
            self.winner = 1
        elif move_count >= 40 and \
                piece_differential < -2:
            self.winner = -1
        elif move_count >= 50 and \
                piece_differential > 3:
            self.winner = 1
        elif move_count >= 50 and \
                piece_differential < -3:
            self.winner = -1
        elif move_count >= 60:
            if piece_differential > 0:
                self.winner = 1
            elif piece_differential < 0:
                self.winner = -1
            else:
                self.winner = 'draw'

    def play(self):

        self.reset()
        player = 1
        move_count = 1
        abort = False

        self.print('======================================================================')
        self.print('Checkers.ai')
        self.print('Author: Chris Larson')
        self.print('Player1: {0}'.format(self.player1))
        self.print('Player2: {0}'.format(self.player2))
        self.print("To forfeit the match, type 'forfeit' when prompted for a move")

        while not any((abort, self.winner)):

            self.print('==================================================================')
            self.print('Move: {}'.format(move_count))
            self.print("Turn: player{}\n".format(1 if player == 1 else 2))
            self.state.print(self.verbose)

            move_available = True
            valid_jumps = self.state.valid_jumps[player]

            while move_available:

                actions_list, probs_list, actions, probs = self.generate_action(player=player)
                if not actions_list and self.winner:
                    break
                else:
                    self.traj[player].update(state=self.state.get_state_view(view=player),
                                             softmax=probs)

                if any((player == 1 and self.player1 == 'human',
                        player == -1 and self.player2 == 'human')) \
                    and len(actions_list[0]) != 2:
                        self.print("Invalid move. Try again")
                        continue

                if valid_jumps:
                    self.print("Available jumps: {}".format(valid_jumps))
                    intersection = list(set(actions_list).intersection(valid_jumps))
                    if intersection:
                        idx = np.argsort([probs_list[actions_list.index(x)] for x in intersection])[::-1]
                        pos_init, pos_final = intersection[idx[0]]
                    else:
                        self.print("{} did not select one of the available jumps, jump forced.".format(
                            "Player1" if player == 1 else "Player2")
                        )
                        idx = np.random.choice(range(len(valid_jumps)), size=1)[0]
                        pos_init, pos_final = valid_jumps[idx]
                    try:
                        self.state.update(pos_init=pos_init,
                                          pos_final=pos_final,
                                          player=player,
                                          move_type='jump')
                    except NotImplementedError:
                        abort = True
                        break
                    valid_jumps = [x for x in self.state.valid_jumps[player] if x[0] == pos_final]
                    if not valid_jumps:
                        move_available = False

                else:
                    move_implemented = False
                    for i, (pos_init, pos_final) in enumerate(actions_list):
                        try:
                            self.state.update(pos_init=pos_init,
                                              pos_final=pos_final,
                                              player=player,
                                              move_type='standard')
                        except NotImplementedError:
                            if any((player == 1 and self.player1 == 'human',
                                    player == -1 and self.player2 == 'human')):
                                self.print("Invalid move. Try again")
                                break
                            else:
                                continue
                        else:
                            move_implemented = True
                            self.print('Move: [%s, %s]' % (pos_init, pos_final))
                            break
                    if not move_implemented and any((player == 1 and type(self.player1) in (Policy, MCTS),
                                                     player == -1 and type(self.player2) in (Policy, MCTS))):
                        self.print("This code is intended to iterate over all moves, state-update NotImplemented")
                        abort = True
                        break
                    elif not move_implemented:
                        continue
                    else:
                        move_available = False

            # Game status
            self.status_check(move_count)
            if abort:
                self.winner = 'null'
                self.aborted_games += 1
            move_count += 1
            player *= -1

        # Print out game stats
        self.print('Ending board:')
        end_board = self.state.get_state_view()
        self.print(end_board.reshape(8, 4))
        n_p1_chkr = len(np.argwhere(end_board == P1_CHKR))
        n_p1_king = len(np.argwhere(end_board == P1_KING))
        n_p2_chkr = len(np.argwhere(end_board == P2_CHKR))
        n_p2_king = len(np.argwhere(end_board == P2_KING))

        if self.winner == 'draw':
            self.print('The game ended in a draw.')
        else:
            self.print('%s wins' % self.player1 if self.winner == 1 else self.player2)
        self.print('Total number of moves: %d' % move_count)
        self.print('Remaining P1 pieces: (checkers: %d, kings: %d)' % (n_p1_chkr, n_p1_king))
        self.print('Remaining P2 pieces: (checkers: %d, kings: %d)' % (n_p2_chkr, n_p2_king))


        if self.winner == 1:
            self.traj[1].update(terminal_rwd=REWARD_WIN)
            self.traj[-1].update(terminal_rwd=REWARD_LOSS)
        if self.winner == -1:
            self.traj[-1].update(terminal_rwd=REWARD_WIN)
            self.traj[1].update(terminal_rwd=REWARD_LOSS)
        elif self.winner == 'draw':
            self.traj[1].update(terminal_rwd=REWARD_DRAW)
            self.traj[-1].update(terminal_rwd=REWARD_DRAW)
        elif self.winner == 'null':
            pass


if __name__ == '__main__':
    TF_CONFIG = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    session = tf.Session(config=TF_CONFIG)
    session2 = tf.Session(config=TF_CONFIG)
    policy = Policy(session=session,
                    load_dir=os.path.join(POLICY_PATH, '81.9acc'),
                    selection='eps-greedy',
                    trainable=False,
                    device="GPU:0")
    policy2 = Policy(session=session2,
                     load_dir=os.path.join(POLICY_PATH, '72acc'),
                     selection='eps-greedy',
                     trainable=False,
                     device="GPU:0")
    session.run(tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ))
    policy.init(session=session)
    policy2.init(session=session2)
    checkers = Game(player1=policy,
                    player2=policy2,
                    verbose=False)
    checkers.play()
    t0 = time.time()
    N = 100
    bar = tqdm(total=N)
    for i in range(N):
        checkers.play()
        bar.update(1)
        bar.set_description("Running simulations: ")
    bar.close()
    t = time.time() - t0
    print("Total time: {}".format(t))
    print("Average game time: {} sec".format(t / N))
    print("aborted_games: {}".format(checkers.aborted_games))

    print(checkers.traj[1].discounted_rewards(step=9))
    print(checkers.traj[-1].discounted_rewards(step=9))