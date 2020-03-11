import copy
from checkers_ai.config import *


class State:

    """
             VIEW_1
           (key: -1)
          00  01  02  03
        04  05  06  07
          08  09  10  11
        12  13  14  15
          16  17  18  19
        20  21  22  23
          24  25  26  27
        28  29  30  31
             VIEW_0
            (key: 1)

         dir3   dir0
             pos
         dir2   dir1
    """
    shell_pieces = {
        EMPTY:   '-   ',
        P2_CHKR: 'x   ',
        P2_KING: 'X   ',
        P1_CHKR: 'o   ',
        P1_KING: 'O   '
    }
    dir_transform = {
        0: 2,
        1: 3,
        2: 0,
        3: 1
    }

    def __init__(self):
        self.init()

    def init(self):
        self.state = copy.deepcopy(BOARD_INIT)
        self.valid_jumps = {1: [], -1: []}

    def print(self, verbose=True):
        if not verbose:
            return
        board = self.state.copy()
        board_str = BOARD_VIEW[0]
        for i, x in enumerate(board):
            if not i % 4:
                board_str += BOARD_VIEW[i // 4 + 1]
            if not i % 8:
                board_str += "  "
            board_str += self.shell_pieces[x]
            if not (i + 1) % 4:
                board_str += "\n"
        board_str += BOARD_VIEW[-1]
        print(board_str)

    async def get_state_view(self, view=1):
        """
        :param view: 1 or -1
        :return: board state from perspective 'view'
        """
        if view == -1:
            return -self.state.copy()[::-1]
        else:
            return self.state.copy()

    @staticmethod
    async def transform_position(pos):
        return 31 - pos

    async def transform_action(self, action):
        a = np.zeros((128,))
        a[action] = 1
        ind = np.argwhere(a.reshape(32, 4) == 1)[0]
        action = np.zeros((32, 4))
        action[await self.transform_position(ind[0]),
               self.dir_transform[ind[1]]] = 1
        return np.argwhere(action.reshape(-1) == 1)[0]

    async def transform_direction(self, direction):
        return self.dir_transform[direction]

    async def get_pieces_and_directions(self, player):
        chkr_dir = P1_CHKR_DIR if player == 1 else P2_CHKR_DIR
        chkr_piece = P1_CHKR if player == 1 else P2_CHKR
        king_piece = P1_KING if player == 1 else P2_KING
        kings_row = P1_KINGS_ROW if player == 1 else P2_KINGS_ROW
        return chkr_piece, king_piece, chkr_dir, KING_DIR, kings_row

    async def __get_jumps(self):
        """
        :returns updated available_jumps member
        """
        for player in (1, -1):
            state = await self.get_state_view()
            valid_jumps = []
            chkr, king, chkr_dir, king_dir, _ = await self.get_pieces_and_directions(player)
            for position in VALID_POS:
                piece = state[position]
                if not piece in (chkr, king):
                    continue
                for direction in king_dir if piece == king else chkr_dir:
                    neighbor = NEIGHBORS[position][direction]
                    next_neighbor = NEXT_NEIGHBORS[position][direction]
                    if neighbor == IV or next_neighbor == IV:
                        continue
                    elif state[next_neighbor] == EMPTY and \
                         state[neighbor] in (-chkr, -king):
                        valid_jumps.append((position, next_neighbor))
            self.valid_jumps[player] = valid_jumps

    @staticmethod
    async def positions2action(pos_init, pos_final):
        """
        :param pos_init: initial position (w.r.t VIEW_0)
        :param pos_final: final position (w.r.t VIEW_0)
        :return: action (32x4 -> posxdir)
        """
        action = np.zeros([32, 4])
        if pos_final in NEXT_NEIGHBORS[pos_init]:
            direction = NEXT_NEIGHBORS.get(pos_init).index(pos_final)
        elif pos_final in NEIGHBORS[pos_init]:
            direction = NEIGHBORS.get(pos_init).index(pos_final)
        else:
            assert False, "{0} not reachable from {1}".format(pos_final, pos_init)
        action[pos_init, direction] = 1
        return action

    async def action2positions(self, action, player):
        """
        :param action: (32x4 -> posxdir) (w.r.t. VIEW_0)
        :param player: player (1 or -1)
        :return: pos_init, pos_final, move_type
        """
        pos, direction = np.argwhere(action.reshape(32, 4) == 1)[0]
        neighbor = NEIGHBORS[pos][direction]
        next_neighbor = NEXT_NEIGHBORS[pos][direction]
        if (pos, next_neighbor) in self.valid_jumps[player]:
            return pos, next_neighbor, 'jump'
        else:
            return pos, neighbor, 'standard'

    async def __set_state(self, state):
        self.state = state
        await self.__get_jumps()

    async def update(self, pos_init, pos_final, player, move_type, set_state=True, verbose=False):
        """
        :param pos_init: initial position (w.r.t. VIEW_0)
        :param pos_final: final_position (w.r.t. VIEW_0)
        :param player: player (1 or -1)
        :param move_type: standard or jump
        :param set_state: update internal state
        :param verbose:
        :return:
        """
        validations = []
        state = await self.get_state_view()
        chkr, king, chkr_dir, king_dir, kings_row = await self.get_pieces_and_directions(player)
        piece = self.state[pos_init]
        valid_dir = chkr_dir if piece == chkr else king_dir
        validations.append(piece in (chkr, king) and state[pos_final] == EMPTY)
        if move_type == 'standard':
            validations.append(pos_final in [NEIGHBORS[pos_init][i] for i in valid_dir])
        elif move_type == 'jump':
            eliminated = int(JUMPS[pos_init, pos_final])
            validations.append(state[eliminated] in (-chkr, -king))
        if all(validations):
            state[pos_final] = state[pos_init]
            state[pos_init] = EMPTY
            if pos_final in kings_row and piece == chkr:
                state[pos_final] = king
            if move_type == 'jump':
                state[eliminated] = EMPTY
                if verbose:
                    print('Position eliminated: %d' % eliminated)
            if set_state:
                await self.__set_state(state)
        else:
            raise NotImplementedError
