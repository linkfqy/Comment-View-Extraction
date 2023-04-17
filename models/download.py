import socket
import requests.packages.urllib3.util.connection as urllib3_cn
 
def allowed_gai_family():
    family = socket.AF_INET    # force IPv4
    return family

urllib3_cn.allowed_gai_family = allowed_gai_family

from huggingface_hub import snapshot_download

snapshot_download("bert-base-chinese",local_dir="./bert-base-chinese",local_dir_use_symlinks=True)
snapshot_download("hfl/chinese-roberta-wwm-ext",local_dir="./chinese-roberta-wwm-ext",local_dir_use_symlinks=True)
snapshot_download("hfl/chinese-roberta-wwm-ext-large",local_dir="./chinese-roberta-wwm-ext-large",local_dir_use_symlinks=True)