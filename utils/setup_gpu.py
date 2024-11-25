import os
import re
import warnings
import torch
from utils.config import Config

def setup_gpu():
    warnings.filterwarnings('ignore')
    torch.manual_seed(Config.SEED)
    
    if Config.USE_GPU:
        def query_gpu(qargs=[]):
            qargs = ['index', 'gpu_name', 'memory.free'] + qargs
            cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
            results = os.popen(cmd).readlines()
            return results

        def select_gpu(results, thres=4096):
            available = []
            try:
                for i, line in enumerate(results):
                    if int(re.findall(r'(.*), (.*?) MiB', line)[0][-1]) > thres:
                        available.append(i)
                return available
            except:
                return ''

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in select_gpu(query_gpu())])

    if not torch.cuda.is_available() and Config.USE_GPU:
        raise RuntimeError("此程式需要 GPU 才能運行！請確認 CUDA 環境是否正確安裝。")

    return torch.device('cuda:0' if torch.cuda.is_available() and Config.USE_GPU else 'cpu')
