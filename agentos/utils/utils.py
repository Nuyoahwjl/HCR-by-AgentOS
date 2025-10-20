# import socket
# import logging
# import psutil
# import subprocess
from openai import OpenAI

# class ColorFormatter(logging.Formatter):
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     RED = '\033[91m'
#     RESET = '\033[0m'

#     def format(self, record):
#         # Apply the appropriate color to the message based on the log level
#         if record.levelno == logging.DEBUG:
#             record.msg = f"{self.RESET}{record.msg}{self.RESET}"
#         elif record.levelno == logging.INFO:
#             record.msg = f"{self.RESET}{record.msg}{self.RESET}"
#         elif record.levelno == logging.WARNING:
#             record.msg = f"{self.YELLOW}{record.msg}{self.RESET}"
#         elif record.levelno == logging.ERROR:
#             record.msg = f"{self.RED}{record.msg}{self.RESET}"
#         elif record.levelno == logging.CRITICAL:
#             record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
#         return super().format(record)


# def get_public_ip():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(('8.8.8.8', 53))
#         ip = s.getsockname()[0]
#         s.close()
#         return ip
#     except Exception as e:
#         print(f"无法获取公共IP地址: {e}")
#         return None
 

# def set_logger(log_level):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(log_level)  
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(log_level)
#     formatter = ColorFormatter('%(asctime)s  - %(message)s')
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
#     return logger
 
# def get_cpu_info():
#     cpu_count = psutil.cpu_count()  
#     cpu_physical_count = psutil.cpu_count(logical=False) 

   
#     memory = psutil.virtual_memory()
#     memory_total = memory.total 
#     memory_available = memory.available 
#     memory_used = memory.used 
#     memory_percent = memory.percent  

#     disk_usage = psutil.disk_usage('/')
#     disk_total = disk_usage.total  
#     disk_used = disk_usage.used  
#     disk_free = disk_usage.free  
#     disk_percent = disk_usage.percent 
    
#     cpu_percent = psutil.cpu_percent(interval=1) 

#     cpu_info ={
#         'cpu_logical_cores': str(cpu_count),
#         'cpu_physical_cores': str(cpu_physical_count),
#         'cpu_usage_percent': str(cpu_percent),
#         'memory_total': str(memory_total),
#         'memory_used': str(memory_used),
#         'memory_available': str(memory_available),
#         'memory_usage_percent': str(memory_percent),      
#         'disk_total': str(disk_total),
#         'disk_used': str(disk_used),
#         'disk_free': str(disk_free),
#         'disk_usage_percent': str(disk_percent)
#     }
    

#     return cpu_info

# def get_gpu_info():
#     try:
#         # 调用nvidia-smi命令
#         smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
#         lines = smi_output.strip().split('\n')
#         gpus = []
#         for line in lines:
#             values = line.split(', ')
#             gpu_info = {
#                 "index": values[0],
#                 "name": values[1],
#                 "utilization": values[2],
#                 "memory_total": values[3],
#                 "memory_free": values[4],
#                 "memory_used": values[5]
#             }
#             gpus.append(gpu_info)
#         return gpus
#     except subprocess.CalledProcessError as e:
#         print("nvidia-smi is not installed or not working properly.")
#         return None

 

##CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model /mnt/7T/xz/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590 --max_num_seqs 1024
# def call_model(messages):
#     # client = OpenAI(
#     #     base_url="http://localhost:8000/v1",
#     #     api_key="xxx",
#     # )

#     # response = client.chat.completions.create(
#     #     model="/mnt/7T/xz/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
#     #     messages = messages
        
#     # )

#     # return response
#     # return completion.choices[0].message.content 

 
    # client = OpenAI(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #     api_key="",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # completion = client.chat.completions.create(
    #     model="qwen-max", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    #     messages=messages
    # )
 
#     return completion.choices[0].message.content 
 
from together import Together
from openai import OpenAI

def call_model(messages, api_key: str | None = None):

    # client = Together(
    #     api_key=api_key,
    #     base_url="https://api.together.xyz/v1",
    # )

    # completion = client.chat.completions.create(
    # model="deepseek-ai/DeepSeek-V3",
    # messages=messages,
    # )

    # return completion.choices[0].message.content 


    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ppinfra.com/openai",
    )

    completion = client.chat.completions.create(
    model="deepseek/deepseek-v3.1",
    messages=messages,
    )

    return completion.choices[0].message.content 