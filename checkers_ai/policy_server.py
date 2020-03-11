import os
import uuid
import time
import asyncio
import base64
import numpy as np
import tensorflow as tf
from aiohttp import web
from checkers_ai.model import Policy
from checkers_ai.state import State


class PolicyServer:

    maxBatchSz = 512
    maxTOQ = 1e-5
    clockSpeed = 1e-6

    def __init__(self, model_path:str, selection:str, url:str, port:int, device:str, gpu_idx=None):
        self.tasks = []
        self.resps = {}

        if device == 'gpu':
            assert gpu_idx is not None
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

        TF_CONFIG = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(config=TF_CONFIG, graph=graph)
            self.policy = Policy(
                session=session,
                load_dir=model_path,
                selection=selection,
                trainable=False,
                device=device
            )
        self.app = self.startServer()
        web.run_app(self.app, host=url, port=port)

    def __del__(self):
        self.app.close()

    async def stateTracker(self):
        t = time.time()
        while True:
            qSize = len(self.tasks)
            qFull = qSize >= self.maxBatchSz
            qWait = (time.time() - t) < self.maxTOQ
            if qSize and (qFull or not qWait):
                if not qFull:
                    t = time.time()
                await self.process()
            else:
                await asyncio.sleep(self.clockSpeed)

    async def enqueue(self, key:str, state:State):
        self.tasks.insert(0, (key, state))

    async def dequeue(self, key):
        result = None
        while not result:
            result = self.resps.pop(key, None)
            if result:
                return result
            else:
                await asyncio.sleep(self.clockSpeed)

    async def requestHandler(self, request):
        response = {
            "action": None,
            "prob": None
        }
        try:
            request = await request.json()
            state = request.get('state')
            if state:
                key = str(uuid.uuid4())
                _ = asyncio.ensure_future(self.enqueue(key, state))
                response = await self.dequeue(key=key)
            return web.json_response(response)
        except Exception as e:
            print("Web handler caught the following exception: {}".format(e))
            return web.json_response(response)

    async def process(self):
        batchSize = min(len(self.tasks), self.maxBatchSz)
        tasks = [self.tasks.pop() for _ in range(batchSize)]
        keys = [task[0] for task in tasks]
        states = np.stack([task[1] for task in tasks], axis=0).reshape(-1, 8, 4).astype(np.int32)
        fetches = [self.policy.probs, self.policy.actions]
        feed_dict = {self.policy.state: states}
        probs, actions = self.policy.session.run(fetches, feed_dict)
        for i in range(batchSize):
            result = {
                "action":  base64.encodebytes(actions[i].dumps()).decode("utf-8"),
                "prob": base64.encodebytes(probs[i].dumps()).decode("utf-8"),
            }
            self.resps.update({keys[i]: result})

    async def startServer(self):
        app = web.Application()
        app.add_routes([web.get("/", self.requestHandler)])
        _ = asyncio.ensure_future(self.stateTracker())
        return app







