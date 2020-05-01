import os
import platform
from requests import get
from tensorflow.python.client import device_lib
from netifaces import interfaces, ifaddresses, AF_INET
for ifaceName in interfaces():
    address = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )][0]
    if "192" in address:
        break

gpu="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

node = platform.node()
system = platform.system()
version = platform.version()
ip = get('https://api.ipify.org').text
devices=device_lib.list_local_devices()
for device in devices:
    if device.device_type=="GPU":
        device_attr=device.physical_device_desc.split(",")
        if gpu in device_attr[0].split(":")[1]:
            gpu_name=device_attr[1].split(":")[1]
file = open(os.path.expanduser("~")+"/system.name")
name=file.readline()
sysmtem_info={"Public IP":ip,"Local IP":address,"Alias":name.rstrip(),"Host":node,"OS":system,"Version":version,"GPU":gpu_name.strip()}
print(sysmtem_info)
