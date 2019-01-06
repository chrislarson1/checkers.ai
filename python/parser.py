import numpy as np
import re
import tqdm
from config import *
from state import State


def parse():
    """
        This program parses OCA_2.pdn, which contains ~22K checkers games, and extracts each of the
        state-action (board-move) pairs into a numpy array and serializes them for python training.

        Example entry from OCA_2.0.pdn:

        '''
            [Event "Manchester 1841"]
            [Date "1841-??-??"]
            [Black "Moorhead, W."]
            [White "Wyllie, J."]
            [Site "Manchester"]
            [Result "0-1"]
            1. 11-15 24-20 2. 8-11 28-24 3. 9-13 22-18 4. 15x22 25x18 5. 4-8 26-22 6. 10-14
            18x9 7. 5x 14 22-18 8. 1-5 18x9 9. 5x14 29-25 10. 11-15 24-19 11. 15x24 25-22 12.
            24-28 22-18 13. 6-9 27-24 14. 8-11 24-19 15. 7-10 20-16 16. 11x20 18-15 17. 2-6
            15-11 18. 12-16 19x12 19. 10-15 11-8 20. 15-18 21-17 21. 13x22 30-26 22. 18x27
            26x17x10x1 0-1
        '''

        Board encoding:

        From OCA_2.0.pdn:
        32 31 30 29
        28 27 26 25
        24 23 22 21
        20 19 18 17
        16 15 14 13
        12 11 10 09
        08 07 06 05
        04 03 02 01

        Board view:
          32  31  30  29
        28  27  26  25
          24  23  22  21
        20  19  18  17
          16  15  14  13
        12  11  10  09
          08  07  06  05
        04  03  02  01


        This program:
        00 01 02 03
        04 05 06 07
        08 09 10 11
        12 13 14 15
        16 17 18 19
        20 21 22 23
        24 25 26 27
        28 29 30 31

        Board view:
          00  01  02  03
        04  05  06  07
          08  09  10  11
        12  13  14  15
          16  17  18  19
        20  21  22  23
          24  25  26  27
        28  29  30  31

        Board transformation: 32 - position

        Model output: 32x4 vector denoting piece to move (col) and direction (row)

        Multiple jumps are treated as separate moves.
        """

    Data, data = [], None
    for line in open(RAW_DFILE, 'r'):
        if any([x in line for x in ('[', ']')]):
            continue
        line = line.strip().lower()
        if line.startswith('1.'):
            data = ""
        elif data is None:
            continue
        if line:
            data += " {}".format(line)
        if not line:
            Data.append(re.sub(r" \{[^]]+\}", "", data.strip().lower()))
            data = None

    def transform_position(pos):
        return 32 - pos

    states, actions = [], []
    bar = tqdm.tqdm(total=len(Data))
    for game in Data:
        state = State()
        game = re.sub(r"\d+\.", " ", game.strip())
        game = re.sub("\s+", " ", game.strip())
        game = game.strip().split(" ")[:-1]
        turn = 1
        invalid = False
        for move in game:
            delim = 'x' if 'x' in move else '-'
            positions = [int(x) for x in move.split(delim)]
            for i in range(len(positions[:-1])):
                x0, x1 = transform_position(positions[i]), \
                         transform_position(positions[i + 1])
                state_input = state.get_state_view(view=turn)
                invalid = state.update(pos_init=x0,
                                       pos_final=x1,
                                       player=turn,
                                       move_type='jump' if delim == 'x' else 'standard')
                if turn == -1:
                    x0, x1 = state.transform_position(x0), state.transform_position(x1)
                action_label = state.positions2action(x0, x1).reshape(-1)
                if not invalid:
                    states.append(state_input)
                    actions.append(action_label)
                else:
                    break
            if invalid:
                break
            turn *= -1
        bar.update(1)
        bar.set_description("Parsing {0}".format(os.path.join(ROOTDIR, RAW_DFILE)))
        del state
    bar.close()
    states = np.stack(states, axis=0).astype(np.int8)
    actions = np.stack(actions, axis=0).astype(np.int8)
    print("Inputs: {0}, Labels: {1}".format(states.shape, actions.shape))
    print("Saving {0} examples from {1} to {2}.".format(len(states), RAW_DFILE, DFILE))
    ind = np.arange(0, len(states))
    np.random.shuffle(ind)
    states, actions = states[ind], actions[ind]
    np.savez(file=os.path.join(ROOTDIR, DFILE), states=states, actions=actions)


if __name__ == '__main__':
    parse()