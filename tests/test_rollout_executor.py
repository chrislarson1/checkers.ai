import os
import time
import asyncio
from checkers_ai.config import POLICY_PATH
from checkers_ai.rollout_executor import (policy_rollout_executor,
                                          policy_rollout_pipe_executor,
                                          policy_rollout_async_executor,
                                          policy_rollout_distributed,
                                          policy_rollout_async_mp_executor)


def test_rollout_executor(device:str, gpu_idx=None):

    num_workers = 3
    games_per_worker = 66
    N = num_workers * games_per_worker
    t = time.time()

    trajs = policy_rollout_executor(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        p1_path=os.path.join(POLICY_PATH, '81.9acc'),
        p2_path=os.path.join(POLICY_PATH, '72acc'),
        p1_selection='eps-greedy',
        p2_selection='eps-greedy',
        p1_device=device,
        p2_device=device,
        p1_gpu_idx=gpu_idx,
        p2_gpu_idx=gpu_idx
    )
    t = time.time() - t

    print("Number of rollouts computed: {}".format(len(trajs)))
    print("Mean rollout compute time: {} sec".format(t / N))


def test_rollout_pipe_executor(device:str, gpu_idx=None):

    num_workers = 2
    games_per_worker = 3
    N = num_workers * games_per_worker
    t = time.time()

    trajs = policy_rollout_pipe_executor(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        p1_path=os.path.join(POLICY_PATH, '81.9acc'),
        p2_path=os.path.join(POLICY_PATH, '72acc'),
        p1_selection='eps-greedy',
        p2_selection='eps-greedy',
        p1_device=device,
        p2_device=device,
        p1_gpu_idx=gpu_idx,
        p2_gpu_idx=gpu_idx
    )

    t = time.time() - t

    print("Number of rollouts computed: {}".format(len(trajs)))
    print("Mean rollout compute time: {} sec".format(t / N))


def test_rollout_async_executor(device:str, gpu_idx=None):
    N = 200
    t = time.time()
    loop = asyncio.get_event_loop()
    trajs = loop.run_until_complete(
        policy_rollout_async_executor(
                num_workers=N,
                p1_path=os.path.join(POLICY_PATH, '81.9acc'),
                p2_path=os.path.join(POLICY_PATH, '72acc'),
                p1_selection='eps-greedy',
                p2_selection='eps-greedy',
                p1_device=device,
                p2_device=device,
                p1_gpu_idx=gpu_idx,
                p2_gpu_idx=gpu_idx
        )
    )
    t = time.time() - t
    loop.close()
    print("Number of rollouts computed: {}".format(len(trajs)))
    print("Mean rollout compute time: {} sec".format(t / N))


def test_rollout_async_mp_executor(device:str, gpu_idx=None):
    num_workers = 3
    games_per_worker = 10
    N = num_workers * games_per_worker
    t = time.time()
    trajs = policy_rollout_async_mp_executor(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        p1_path=os.path.join(POLICY_PATH, '81.9acc'),
        p2_path=os.path.join(POLICY_PATH, '72acc'),
        p1_selection='eps-greedy',
        p2_selection='eps-greedy',
        p1_device=device,
        p2_device=device,
        p1_gpu_idx=gpu_idx,
        p2_gpu_idx=gpu_idx
    )
    t = time.time() - t
    print("Number of rollouts computed: {}".format(len(trajs)))
    print("Mean rollout compute time: {} sec".format(t / N))


def test_rollout_distributed_executor(device:str, gpu_idx=None):
    num_workers = 10
    games_per_worker = 50
    N = num_workers * games_per_worker
    t = time.time()
    trajs = policy_rollout_distributed(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        p1_url="http://0.0.0.0:8081",
        p2_url="http://0.0.0.0:8081",
        p1_selection='eps-greedy',
        p2_selection='eps-greedy',
        p1_device=device,
        p2_device=device,
        p1_gpu_idx=gpu_idx,
        p2_gpu_idx=gpu_idx
    )
    t = time.time() - t
    print("Number of rollouts computed: {}".format(len(trajs)))
    print("Mean rollout compute time: {} sec".format(t / N))

if __name__ == '__main__':
    # test_rollout_executor(device='cpu', gpu_idx=None)
    # test_rollout_pipe_executor(device='cpu', gpu_idx=None)
    # test_rollout_async_executor(device='cpu', gpu_idx=None)
    # test_rollout_async_mp_executor(device='cpu', gpu_idx=None)
    test_rollout_distributed_executor(device='cpu', gpu_idx=None)
