from checkers_ai.config import *
from checkers_ai.state import State
from checkers_ai.trajectory import Trajectory
from checkers_ai.player import HumanPlayer


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

    def __init__(self, player1, player2, verbose=False):
        self.state = State()
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose
        self.winner = None
        self.traj = {}
        self.aborted_games = 0
        self.reset()

    def __del__(self):
        del self.player1
        del self.player2

    @property
    def trajectory(self):
        return self.traj

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

    def play(self, first_move:int):

        """
        :param first_move: 1: player1
                          -1: player2
        """

        self.reset()
        player = first_move
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

            contestant = self.player1 if player == 1 else self.player2
            move_available = True
            valid_jumps = self.state.valid_jumps[player]

            while move_available:

                # actions_list, probs_list, actions, probs = self.generate_action(player=player)
                actions_list, probs_list, actions, probs, self.winner = contestant.get_action(self.state, player)

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
                    if not move_implemented and any((player == 1 and not isinstance(self.player1, HumanPlayer),
                                                     player == -1 and not isinstance(self.player2, HumanPlayer))):
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
            else:
                move_count += 1
                player *= -1

        # Print out game stats
        end_board = self.state.get_state_view()
        n_p1_chkr = sum(end_board == P1_CHKR)
        n_p1_king = sum(end_board == P1_KING)
        n_p2_chkr = sum(end_board == P2_CHKR)
        n_p2_king = sum(end_board == P2_KING)
        self.print('Ending board:')
        self.print(end_board.reshape(8, 4))
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
            self.traj[1].update(terminal_rwd=REWARD_LOSS)
            self.traj[-1].update(terminal_rwd=REWARD_WIN)
        elif self.winner == 'draw':
            self.traj[1].update(terminal_rwd=REWARD_DRAW)
            self.traj[-1].update(terminal_rwd=REWARD_DRAW)
        elif self.winner == 'null':
            pass


class GameAsync:
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

    def __init__(self, player1, player2, verbose=False):
        self.state = State()
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose
        self.winner = None
        self.traj = {}
        self.aborted_games = 0
        self.reset()

    def __del__(self):
        del self.player1
        del self.player2

    @property
    def trajectory(self):
        return self.traj

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

    async def status_check(self, move_count):
        n_p1 = len(np.argwhere(await self.state.get_state_view() > EMPTY))
        n_p2 = len(np.argwhere(await self.state.get_state_view() < EMPTY))
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

    async def play(self, first_move: int):

        """
        :param first_move: 1: player1
                          -1: player2
        """

        self.reset()
        player = first_move
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

            contestant = self.player1 if player == 1 else self.player2
            move_available = True
            valid_jumps = self.state.valid_jumps[player]

            while move_available:

                actions_list, probs_list, actions, probs, self.winner = await contestant.get_action(self.state, player)

                if not actions_list and self.winner:
                    break
                else:
                    self.traj[player].update(state=await self.state.get_state_view(view=player),
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
                        await self.state.update(pos_init=pos_init,
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
                            await self.state.update(pos_init=pos_init,
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
                    if not move_implemented and any((player == 1 and not isinstance(self.player1, HumanPlayer),
                                                     player == -1 and not isinstance(self.player2, HumanPlayer))):
                        self.print("This code is intended to iterate over all moves, state-update NotImplemented")
                        abort = True
                        break
                    elif not move_implemented:
                        continue
                    else:
                        move_available = False

            # Game status
            await self.status_check(move_count)
            if abort:
                self.winner = 'null'
                self.aborted_games += 1
            else:
                move_count += 1
                player *= -1

        # Print out game stats
        end_board = await self.state.get_state_view()
        n_p1_chkr = sum(end_board == P1_CHKR)
        n_p1_king = sum(end_board == P1_KING)
        n_p2_chkr = sum(end_board == P2_CHKR)
        n_p2_king = sum(end_board == P2_KING)
        self.print('Ending board:')
        self.print(end_board.reshape(8, 4))
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
            self.traj[1].update(terminal_rwd=REWARD_LOSS)
            self.traj[-1].update(terminal_rwd=REWARD_WIN)
        elif self.winner == 'draw':
            self.traj[1].update(terminal_rwd=REWARD_DRAW)
            self.traj[-1].update(terminal_rwd=REWARD_DRAW)
        elif self.winner == 'null':
            pass

        return self.traj
