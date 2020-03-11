import asyncio
from checkers_ai.config import *
from checkers_ai.game import Game, GameAsync
from checkers_ai.player import PolicyPlayer, DistributedPolicyPlayer
from multiprocessing import Process, Pipe, cpu_count

""" 
    Parallellized rollout training:
    
        * get N_cpu, N_gpu
        * k = N_cpu_worker_node per gpu_compute_node
        
        * new MasterTrainer in main process
        * new MasterQueue in main process

        * for gpu in range(N_gpu - 1):
            * new PolicyServer w/MasterQueue access
            * new SubQueue
            * for cpu in range(k):
                * new process w/game functor, SubQueue access, Pipe connection to PolicyServer
        
        * while training:
        
            * for step in steps_per_model_change:
                * execute rollouts on all N_gpu sub-nodes, collect trajectories via MasterQueue
                * execute gradient udpate
                
            * Cache model

"""


def policy_rollout_executor(num_workers:int, games_per_worker:int, **kwargs):

    assert num_workers < cpu_count(), 'num_cores must be < cpu_cores!'

    def run(pipe):

        player1 = PolicyPlayer(
            load_dir=kwargs.get('p1_path'),
            selection=kwargs.get('p1_selection'),
            device=kwargs.get('p1_device'),
            gpu_idx=kwargs.get('p1_gpu_idx')
        )

        player2 = PolicyPlayer(
            load_dir=kwargs.get('p1_path'),
            selection=kwargs.get('p1_selection'),
            device=kwargs.get('p1_device'),
            gpu_idx=kwargs.get('p2_gpu_idx')
        )

        game = Game(player1=player1, player2=player2, verbose=False)
        output = []
        for _ in range(games_per_worker):
            game.play(first_move=np.random.choice([-1, 1]))
            output.append(game.traj)

        # Pipe trajectories back to master process
        pipe.send(output)
        pipe.close()

        # Garbage collection
        del game, player1, player2

    processes = []
    connections = []
    trajectories = []

    for _ in range(num_workers):
        parent_conn, child_conn = Pipe()
        proc = Process(target=run, args=(child_conn,))
        proc.start()
        processes.append(proc)
        connections.append(parent_conn)
        # print('proc spawned')

    while all((processes, connections)):
        for i, (proc, conn) in enumerate(zip(processes, connections)):
            if conn.poll():
                trajectories.extend(conn.recv())
                processes.pop(i).join()
                connections.pop(i).close()
                # print("proc joined")
                break

    return trajectories


def policy_rollout_pipe_executor(num_workers:int, games_per_worker:int, **kwargs):

    assert num_workers < cpu_count(), 'batch_size must be < cpu cores!'

    def run(pipe):
        game = pipe.recv()
        output = []
        for _ in range(games_per_worker):
            game.play(first_move=np.random.choice([-1, 1]))
            output.append(game.trajectory)
        pipe.send(output)
        pipe.close()
        del game

    # Policy 1
    player1 = PolicyPlayer(
        load_dir=kwargs.get('p1_path'),
        selection=kwargs.get('p1_selection'),
        device=kwargs.get('p1_device'),
        gpu_idx=kwargs.get('p1_gpu_idx')
    )

    # Policy 2
    player2 = PolicyPlayer(
        load_dir=kwargs.get('p1_path'),
        selection=kwargs.get('p1_selection'),
        device=kwargs.get('p1_device'),
        gpu_idx=kwargs.get('p2_gpu_idx')
    )

    # Simulation
    game = Game(player1=player1, player2=player2, verbose=False)

    processes = []
    connections = []
    trajectories = []

    for _ in range(num_workers):
        parent_conn, child_conn = Pipe()
        parent_conn.send(game)
        proc = Process(target=run, args=(child_conn,))
        proc.start()
        processes.append(proc)
        connections.append(parent_conn)
        # print('proc spawned')

    while all((processes, connections)):
        for i, (proc, conn) in enumerate(zip(processes, connections)):
            if conn.poll():
                trajectories.extend(conn.recv())
                processes.pop(i).join()
                connections.pop(i).close()
                # print("proc joined")
                break

    return trajectories


async def policy_rollout_async_executor(num_workers:int, **kwargs):
    player1 = PolicyPlayer(
        load_dir=kwargs.get('p1_path'),
        selection=kwargs.get('p1_selection'),
        device=kwargs.get('p1_device'),
        gpu_idx=kwargs.get('p1_gpu_idx')
    )
    player2 = PolicyPlayer(
        load_dir=kwargs.get('p1_path'),
        selection=kwargs.get('p1_selection'),
        device=kwargs.get('p1_device'),
        gpu_idx=kwargs.get('p2_gpu_idx')
    )
    game = GameAsync(player1=player1, player2=player2, verbose=False)
    tasks = []
    for _ in range(num_workers):
        tasks.append(asyncio.ensure_future(game.play(first_move=np.random.choice([-1, 1]))))
    trajs = await asyncio.gather(*tasks)
    return trajs


async def exec_async(p1, p2, N):
    tasks = []
    for _ in range(N):
        game = GameAsync(player1=p1, player2=p2, verbose=False)
        tasks.append(asyncio.ensure_future(game.play(first_move=np.random.choice([-1, 1]))))
    trajs = await asyncio.gather(*tasks)
    return trajs


async def close_session(player):
    await player.session.close()


def policy_rollout_async_mp_executor(num_workers:int, games_per_worker:int, **kwargs):

    assert num_workers < cpu_count(), 'num_cores must be < cpu_cores!'

    def run(pipe):
        player1 = PolicyPlayer(
            load_dir=kwargs.get('p1_path'),
            selection=kwargs.get('p1_selection'),
            device=kwargs.get('p1_device'),
            gpu_idx=kwargs.get('p1_gpu_idx')
        )
        player2 = PolicyPlayer(
            load_dir=kwargs.get('p1_path'),
            selection=kwargs.get('p1_selection'),
            device=kwargs.get('p1_device'),
            gpu_idx=kwargs.get('p2_gpu_idx')
        )
        game = GameAsync(player1=player1, player2=player2, verbose=False)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(exec_async(game, games_per_worker))
        trajs = game.traj
        loop.close()
        pipe.send(trajs)
        pipe.close()
        del game, player1, player2

    processes = []
    connections = []
    trajectories = []

    for _ in range(num_workers):
        parent_conn, child_conn = Pipe()
        proc = Process(target=run, args=(child_conn,))
        proc.start()
        processes.append(proc)
        connections.append(parent_conn)
        # print('proc spawned')

    while all((processes, connections)):
        for i, (proc, conn) in enumerate(zip(processes, connections)):
            if conn.poll():
                trajectories.extend(conn.recv())
                processes.pop(i).join()
                connections.pop(i).close()
                # print("proc joined")
                break

    return trajectories


def policy_rollout_distributed(num_workers:int, games_per_worker:int, **kwargs):

    # assert num_workers < (cpu_count() - 1), 'num_cores must be < cpu_cores!'

    def run(pipe):
        p1 = DistributedPolicyPlayer(
            url=kwargs.get('p1_url'),
            selection=kwargs.get('p1_selection')
        )
        p2 = DistributedPolicyPlayer(
            url=kwargs.get('p2_url'),
            selection=kwargs.get('p2_selection')
        )
        loop = asyncio.get_event_loop()
        trajs = loop.run_until_complete(exec_async(p1, p2, games_per_worker))
        loop.run_until_complete(close_session(p1))
        loop.run_until_complete(close_session(p2))
        loop.close()
        pipe.send(trajs)
        pipe.close()

    processes = []
    connections = []
    trajectories = []

    for _ in range(num_workers):
        parent_conn, child_conn = Pipe()
        proc = Process(target=run, args=(child_conn,))
        proc.start()
        processes.append(proc)
        connections.append(parent_conn)
        # print('proc spawned')

    while all((processes, connections)):
        for i, (proc, conn) in enumerate(zip(processes, connections)):
            if conn.poll():
                trajectories.extend(conn.recv())
                processes.pop(i).join()
                connections.pop(i).close()
                # print("proc joined")
                break

    # del processes, connections

    return trajectories

