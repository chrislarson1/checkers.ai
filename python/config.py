import os
import sys
import json
import numpy as np
import tensorflow as tf


#######################################################################
############## ENVIRONMENT VARIABLES DEFINED HERE #####################
#######################################################################
ROOTDIR = "/media/chris/HDD/checkers.ai"
RAW_DFILE = os.path.join(ROOTDIR, 'data/OCA_2.0.pdn')
DFILE = os.path.join(ROOTDIR, 'data/OCA_2.0.npz')
JUMPS_FILE = os.path.join(ROOTDIR, 'data/jumps.csv')
SKIPS_IDX_FILE = os.path.join(ROOTDIR, 'data/skips_indexed.csv')
POLICY_PATH = os.path.join(ROOTDIR, 'params/policy')
VALUE_PATH = os.path.join(ROOTDIR, 'params/value')

#######################################################################
######################## CHECKERS GAME CONFIG #########################
#######################################################################
BOARD_VIEW = [
    "\t    PLAYER2        |       PLAYER2\n",
    "\t  00  01  02  03   |   ",
    "\t04  05  06  07     |   ",
    "\t  08  09  10  11   |   ",
    "\t12  13  14  15     |   ",
    "\t  16  17  18  19   |   ",
    "\t20  21  22  23     |   ",
    "\t  24  25  26  27   |   ",
    "\t28  29  30  31     |   ",
    "\t    PLAYER1        |       PLAYER1\n"
]
P1_CHKR = 1
P1_KING = 3
P1_CHKR_DIR = [0, 3]
P2_CHKR = -P1_CHKR
P2_KING = -P1_KING
P2_CHKR_DIR = [1, 2]
KING_DIR = [0, 1, 2, 3]
EMPTY = 0
ODD_POS = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
EVEN_POS = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
P1_KINGS_ROW = [0, 1, 2, 3]
P2_KINGS_ROW = [28, 29, 30, 31]
VALID_POS = range(32)
BOARD_INIT = np.array([
    [P2_CHKR, P2_CHKR, P2_CHKR, P2_CHKR],
    [P2_CHKR, P2_CHKR, P2_CHKR, P2_CHKR],
    [P2_CHKR, P2_CHKR, P2_CHKR, P2_CHKR],
    [ EMPTY,   EMPTY,   EMPTY,   EMPTY ],
    [ EMPTY,   EMPTY,   EMPTY,   EMPTY ],
    [P1_CHKR, P1_CHKR, P1_CHKR, P1_CHKR],
    [P1_CHKR, P1_CHKR, P1_CHKR, P1_CHKR],
    [P1_CHKR, P1_CHKR, P1_CHKR, P1_CHKR]
]).reshape(-1).astype(np.int32)
JUMPS = np.genfromtxt(fname=JUMPS_FILE, delimiter=',')
MAX_MOVES = 60
SELECT_EPS = 0.5
SELECT_N = 128
REWARD_WIN = float(MAX_MOVES) / 2
REWARD_LOSS = -REWARD_WIN
REWARD_DRAW = 5.0
GAMMA = 0.99
VALUE_RANGE = [-53, 53]
IV = ''
NEIGHBORS = {0: [IV, 5, 4, IV],
             1: [IV, 6, 5, IV],
             2: [IV, 7, 6, IV],
             3: [IV, IV, 7, IV],
             4: [0, 8, IV, IV],
             5: [1, 9, 8, 0],
             6: [2, 10, 9, 1],
             7: [3, 11, 10, 2],
             8: [5, 13, 12, 4],
             9: [6, 14, 13, 5],
             10: [7, 15, 14, 6],
             11: [IV, IV, 15, 7],
             12: [8, 16, IV, IV],
             13: [9, 17, 16, 8],
             14: [10, 18, 17, 9],
             15: [11, 19, 18, 10],
             16: [13, 21, 20, 12],
             17: [14, 22, 21, 13],
             18: [15, 23, 22, 14],
             19: [IV, IV, 23, 15],
             20: [16, 24, IV, IV],
             21: [17, 25, 24, 16],
             22: [18, 26, 25, 17],
             23: [19, 27, 26, 18],
             24: [21, 29, 28, 20],
             25: [22, 30, 29, 21],
             26: [23, 31, 30, 22],
             27: [IV, IV, 31, 23],
             28: [24, IV, IV, IV],
             29: [25, IV, IV, 24],
             30: [26, IV, IV, 25],
             31: [27, IV, IV, 26]}

NEXT_NEIGHBORS = {0: [IV, 9, IV, IV],
                  1: [IV, 10, 8, IV],
                  2: [IV, 11, 9, IV],
                  3: [IV, IV, 10, IV],
                  4: [IV, 13, IV, IV],
                  5: [IV, 14, 12, IV],
                  6: [IV, 15, 13, IV],
                  7: [IV, IV, 14, IV],
                  8: [1, 17, IV, IV],
                  9: [2, 18, 16, 0],
                  10: [3, 19, 17, 1],
                  11: [IV, IV, 18, 2],
                  12: [5, 21, IV, IV],
                  13: [6, 22, 20, 4],
                  14: [7, 23, 21, 5],
                  15: [IV, IV, 22, 6],
                  16: [9, 25, IV, IV],
                  17: [10, 26, 24, 8],
                  18: [11, 27, 25, 9],
                  19: [IV, IV, 26, 10],
                  20: [13, 29, IV, IV],
                  21: [14, 30, 28, 12],
                  22: [15, 31, 29, 13],
                  23: [IV, IV, 30, 14],
                  24: [17, IV, IV, IV],
                  25: [18, IV, IV, 16],
                  26: [19, IV, IV, 17],
                  27: [IV, IV, IV, 18],
                  28: [21, IV, IV, IV],
                  29: [22, IV, IV, 20],
                  30: [23, IV, IV, 21],
                  31: [IV, IV, IV, 22]}


#######################################################################
########################## Model Parameters ###########################
#######################################################################
PARAM_INIT = tf.contrib.layers.xavier_initializer()

INCEP_ACT = tf.nn.tanh

HWY_LAYERS = 3

KEEP_PROB = 0.67

LRATE = 0.001

PASS_ANNEAL_RATE = 0.99

FAIL_ANNEAL_RATE = 0.9

MIN_LRATE = 1e-7

LAMBDA = 1e-7

# KERNEL_SIZES = [
#     ((1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)),
#     ((1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)),
#     ((1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4))
# ]
#
# N_KERNELS = [
#     [20, 20, 20, 20, 20, 20, 20, 20],
#     [20, 20, 20, 20, 20, 20, 20, 20],
#     [20, 20, 20, 20, 20, 20, 20, 20]
# ]

KERNEL_SIZES = [
    ((1, 1), (2, 2), (3, 3), (4, 4)),
    ((1, 1), (2, 2), (3, 3), (4, 4)),
    ((1, 1), (2, 2), (3, 3), (4, 4)),
    ((1, 1), (2, 2), (3, 3), (4, 4))
]

NK = 24

N_KERNELS = [
    [NK, NK, NK, NK],
    [NK, NK, NK, NK],
    [NK, NK, NK, NK],
    [NK, NK, NK, NK]
]

HWY_BIAS = -2.0


def filetree():
    for x in (POLICY_PATH, VALUE_PATH):
        if not os.path.isdir(os.path.join(x, 'tmp')):
            os.makedirs(x)



if __name__ == '__main__':
    filetree()