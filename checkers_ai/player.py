import aiohttp
import base64
import pickle
import tensorflow as tf
from checkers_ai.config import *
from checkers_ai.model import Policy, Value
from checkers_ai.state import State


class HumanPlayer:

    def __init__(self):
        pass

    @staticmethod
    def get_action(player:int):

        actions_list, probs_list, actions, probs, winner = [], [], [], [], None

        move_invalid = True
        while move_invalid:

            move = input("Enter move as 'pos_init pos_final':")

            if move.lower() == 'forfeit':
                winner = -player
            elif move == 'draw':
                winner = 'draw'
            else:
                positions = tuple([int(x) for x in move.split(' ')])
                if len(positions) != 2 or not all([x in range(32) for x in positions]):
                    print("Invalid move. Try again")
                    continue
                actions_list.append(positions)
                probs_list.append(1.0)
                move_invalid = False

        return actions_list, probs_list, actions, probs, winner


class PolicyPlayer:

    def __init__(self, load_dir:str, selection:str, device:str, gpu_idx=None):

        self.selection = selection

        if device == 'gpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

        TF_CONFIG = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        graph = tf.Graph()

        with graph.as_default():

            session = tf.Session(config=TF_CONFIG, graph=graph)

            self.policy = Policy(
                session=session,
                load_dir=load_dir,
                selection=selection,
                trainable=False,
                device=device
            )

    async def get_action(self, state:State, player:int):
        actions_list, probs_list, actions, probs = [], [], [], []
        board_state = await state.get_state_view(view=player)
        feed_dict = {
            self.policy.state: board_state.reshape(1, 8, 4).astype(np.int32),
        }
        probs, actions = self.policy.session.run(
            fetches=[self.policy.probs, self.policy.actions],
            feed_dict=feed_dict
        )
        probs, actions = probs[0, :SELECT_N], actions[0, :SELECT_N]
        order = np.arange(0, SELECT_N)
        if self.selection == 'multinomial':
            order = np.random.choice(order,
                                     size=SELECT_N,
                                     replace=False,
                                     p=(probs + 1e-6) / np.sum(probs + 1e-6))

        elif all((self.selection == 'eps-greedy', np.random.random() <= SELECT_EPS)):
            order = np.random.choice(order,
                                     size=SELECT_N,
                                     replace=False,
                                     p=(probs + 1e-6) / np.sum(probs + 1e-6))
        for i, (prob, action) in enumerate(zip(probs[order], actions[order])):
            move = np.zeros((128,))
            action = action if player == 1 else await state.transform_action(action)
            move[action] = 1
            pos_init, pos_final, move_type = await state.action2positions(move, player=player)
            if pos_init and pos_final:
                actions_list.append((pos_init, pos_final))
                probs_list.append(prob)
        return actions_list, probs_list, actions, probs, None


# class DistributedPolicyPlayer:
#
#     def __init__(self, uid:str, queue:Queue, child_conn, selection:str):
#         self.uid = uid
#         self.queue = queue
#         self.conn = child_conn
#         self.selection = selection
#
#     def select(self, player, state, actions, probs):
#
#         actions_list, probs_list = [], []
#
#         probs, actions = probs[0, :SELECT_N], actions[0, :SELECT_N]
#         order = np.arange(0, SELECT_N)
#
#         if self.selection == 'multinomial':
#             order = np.random.choice(order,
#                                      size=SELECT_N,
#                                      replace=False,
#                                      p=(probs + 1e-6) / np.sum(probs + 1e-6))
#
#         elif all((self.selection == 'eps-greedy', np.random.random() <= SELECT_EPS)):
#             order = np.random.choice(order,
#                                      size=SELECT_N,
#                                      replace=False,
#                                      p=(probs + 1e-6) / np.sum(probs + 1e-6))
#
#         for i, (prob, action) in enumerate(zip(probs[order], actions[order])):
#             move = np.zeros((128,))
#             action = action if player == 1 else state.transform_action(action)
#             move[action] = 1
#             pos_init, pos_final, move_type = state.action2positions(move, player=player)
#             if pos_init and pos_final:
#                 actions_list.append((pos_init, pos_final))
#                 probs_list.append(prob)
#
#         return actions_list, probs_list, actions, probs, None
#
#     def get_action(self, state:State, player:int):
#
#         payload = {
#             "id": self.uid,
#             "state": state.get_state_view(view=player)
#         }
#
#         self.queue.put_nowait(payload)
#
#         while True:
#             if self.conn.poll():
#                 probs, actions = self.conn.recv()
#                 return self.select(player, state, actions, probs)


class DistributedPolicyPlayer:

    def __init__(self, url:str, selection:str):
        self.url = url
        self.selection = selection
        self.session = aiohttp.ClientSession()

    async def get_action(self, state:State, player:int):
        board_state = await state.get_state_view(view=player)
        payload = {"state": board_state.tolist()}
        async with self.session.get(self.url, json=payload) as resp:
            result = await resp.json()
        if not all((result.get("prob"), result.get("action"))):
            return None, None, None, None, None
        probs = pickle.loads(base64.b64decode(result.get("prob"))).astype(np.float32).reshape(1, -1)
        actions = pickle.loads(base64.b64decode(result.get("action"))).reshape(1, -1)
        probs, actions = probs[0, :SELECT_N], actions[0, :SELECT_N]
        order = np.arange(0, SELECT_N)
        if self.selection == 'multinomial':
            order = np.random.choice(order,
                                     size=SELECT_N,
                                     replace=False,
                                     p=(probs + 1e-6) / np.sum(probs + 1e-6))
        elif all((self.selection == 'eps-greedy', np.random.random() <= SELECT_EPS)):
            order = np.random.choice(order,
                                     size=SELECT_N,
                                     replace=False,
                                     p=(probs + 1e-6) / np.sum(probs + 1e-6))
        actions_list, probs_list = [], []
        for i, (prob, action) in enumerate(zip(probs[order], actions[order])):
            move = np.zeros((128,))
            action = action if player == 1 else await state.transform_action(action)
            move[action] = 1
            pos_init, pos_final, move_type = await state.action2positions(move, player=player)
            if pos_init and pos_final:
                actions_list.append((pos_init, pos_final))
                probs_list.append(prob)
        # print(actions_list[0], probs_list[0])
        return actions_list, probs_list, actions, probs, None


class MCTSPlayer:

    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_action(self, state):
        pass
