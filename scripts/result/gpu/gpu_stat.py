#!/usr/bin/env python                                                                                                                     
# gpu_stat.py [DELAY [COUNT]]                                                                                                             
# dump some gpu stats as a line of json                                                                                                   
# {"util":{"PCIe":"0", "memory":"11", "video":"0", "graphics":"13"}, "used_mem":"161"}                                                    

import json, socket, subprocess, sys, time
try:
    delay = int(sys.argv[1])
except:
    delay = 1

try:
    count = int(sys.argv[2])
except:
    count = None

i = 0
while True:
    GPU = socket.gethostname() + ":0[gpu:0]"
    cmd = ['nvidia-settings', '-t', '-q', GPU + '/GPUUtilization']
    gpu_util = subprocess.check_output(cmd).strip().split(",")
    gpu_util = dict([f.strip().split("=") for f in gpu_util])
    cmd[-1] = GPU + '/UsedDedicatedGPUMemory'
    gpu_used_mem = subprocess.check_output(cmd).strip()
    print json.dumps({"used_mem": gpu_used_mem, "util": gpu_util, "time": int(time.time())})
    if count != None:
        i += 1
        if i == count:
            exit(0)
    time.sleep(delay)