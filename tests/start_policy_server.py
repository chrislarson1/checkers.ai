import os
from checkers_ai.config import POLICY_PATH
from checkers_ai.policy_server import PolicyServer


def start_server(model_path:str, url:str, port:int, device:str, gpu_idx=None):
    PolicyServer(
        model_path=model_path,
        selection='eps-greedy',
        device=device,
        gpu_idx=gpu_idx,
        url=url,
        port=port
    )


if __name__ == '__main__':
    url = '0.0.0.0'
    port = 8081
    model_path = os.path.join(POLICY_PATH, '81.9acc')
    start_server(model_path, url, port, device='gpu', gpu_idx=0)
