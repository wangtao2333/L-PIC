import torch, queue
import numpy as np
from pprint import pprint
import os, time, psutil, threading, gc
from torchvision import models
import multiprocessing as mp
##  模型加载流水线函数
def load_model(model_buffer, load_buffer, wait_model):
    while True:
        idx = load_buffer.get()
        model_buffer[idx] = torch.load('resnet152_layers/layer' + str(idx) + '.bin').eval()
        wait_model.release()
            
## 删除模型
def delete_model(model_buffer, del_buffer):
    while True:
        del model_buffer[del_buffer.get()]

def resnet_mem(resolution = 608, nums = 57):
    shape0 = resolution * resolution * 3
    channel = 3
    out_shape = []
    for i in range(57):
        a = torch.load('resnet152_layers/layer' + str(i) + '.bin')
        if hasattr(a, 'downsample'):
            if any(isinstance(module, torch.nn.Sequential) for module in a.modules()):
                temp = resolution * resolution * channel
                resolution = (resolution - a.downsample[0].kernel_size[0]) // a.downsample[0].stride[0] + 1
                shape0 = resolution * resolution * a.downsample[0].out_channels
                out_shape.append(shape0 + (temp << 1))
            else:
                out_shape.append(shape0 * 3)
            channel = a.conv3.out_channels
        elif isinstance(a, torch.nn.Conv2d):
            temp = resolution * resolution * channel
            resolution = (resolution - a.kernel_size[0] + (a.padding[0] << 1)) // a.stride[0] + 1
            shape0 = resolution * resolution * a.out_channels
            out_shape.append(shape0 + temp)
            channel = a.out_channels
        elif isinstance(a, torch.nn.MaxPool2d):
            out_shape.append(shape0 << 1)
            temp = resolution * resolution * channel
            resolution = (resolution - a.kernel_size + (a.padding << 1)) // a.stride + 1
            shape0 = resolution * resolution * channel
            # out_shape.append(temp << 1)
        elif isinstance(a, torch.nn.AdaptiveAvgPool2d):
            out_shape.append(channel + resolution * resolution * channel)
            shape0 = channel
        elif isinstance(a, torch.nn.Linear):
            out_shape.append(a.in_features << 1)
            shape0 = a.out_features
        else:
            out_shape.append(shape0 << 1)
    for i in range(nums):
        out_shape[i] = out_shape[i] << 2
    return out_shape
    

def get_filesize(model):
    layer_nums = 0
    if model == 'resnet152':
        layer_nums = 57
    path = model + '_layers'
    file_size = []
    for layer in range(layer_nums):
        file_size.append(os.path.getsize(path + '/layer' + str(layer) + '.bin'))
    return file_size
    
def no_block(resolution, child):
    start = time.time()
    layers_num, batch_size, task_num = 57, 1, 10
    model_buffer = {}
    buffer_size = 3
    load_buffer, del_buffer = queue.Queue(), queue.Queue()
    wait_model = threading.Semaphore(0)
    loading = threading.Thread(target=load_model, args=(model_buffer, load_buffer, wait_model))
    deleting = threading.Thread(target=delete_model, args=(model_buffer, del_buffer))
    loading.daemon = True
    deleting.daemon = True
    loading.start()
    deleting.start()
    load_buffer_ptr = 0
    del_buffer_ptr = 0
    for i in range(buffer_size):
        load_buffer.put(load_buffer_ptr)
        load_buffer_ptr += 1
    torch.set_num_threads(1)
    print('init:', psutil.Process(os.getpid()).memory_info().rss >> 20)
    for i in range(task_num):
        input = torch.randn(batch_size, 3, resolution, resolution)
        with torch.no_grad():
            for layer in range(layers_num):
                load_buffer.put(load_buffer_ptr)
                load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                wait_model.acquire()
                input = model_buffer[layer](input)
                del_buffer.put(del_buffer_ptr)
                del_buffer_ptr = (del_buffer_ptr + 1) % layers_num
    child.send(True)
    print('resolution --', resolution, ':', time.time() - start)
    print('-------------------------------------------------------------------')

##  拆分测试函数
def Inference(resolution, main_pid, max_resolution, block_lock, block_nums):
    sef_pid = os.getpid()
    process = psutil.Process(sef_pid)
    file_size = get_filesize('resnet152')
    out_shape = resnet_mem(resolution=resolution)
    layers_num, batch_size, task_num = 57, 1, 20
    model_buffer = {}
    buffer_size = 3
    load_buffer, del_buffer = queue.Queue(), queue.Queue()
    wait_model = threading.Semaphore(0)
    loading = threading.Thread(target=load_model, args=(model_buffer, load_buffer, wait_model))
    deleting = threading.Thread(target=delete_model, args=(model_buffer, del_buffer))
    loading.daemon = True
    deleting.daemon = True
    loading.start()
    deleting.start()
    load_buffer_ptr = 0
    del_buffer_ptr = 0
    for i in range(buffer_size):
        load_buffer.put(load_buffer_ptr)
        load_buffer_ptr += 1
    max_mem = process.memory_info().rss
    threshold = 1700 << 20 #global_memory(main_process) + (100 << 21)
    print('init:',max_mem >> 20)
    torch.set_num_threads(1)
    print(psutil.virtual_memory().used >> 20)
    start = time.time()
    for i in range(task_num):
        input = torch.randn(batch_size, 3, resolution, resolution)
        with torch.no_grad():
            for layer in range(layers_num):
                rest = threshold - out_shape[layer] * batch_size
                if wait_model._value == 0:
                    rest -= file_size[load_buffer_ptr]
                if psutil.virtual_memory().used > rest:
                    if layer < 6 and layer > 2:                        #wait
                        with block_lock:
                            block_nums.value += 1
                        if resolution == max_resolution.value:
                            while(psutil.virtual_memory().used >= rest):
                                if block_nums.value == len(psutil.Process(main_pid).children()):
                                    break
                                time.sleep(0.1)
                        else:
                            while(psutil.virtual_memory().used >= rest):
                                if wait_model._value > 1:
                                    load_buffer_ptr = (load_buffer_ptr - 1) % layers_num
                                    del_buffer.put(load_buffer_ptr)
                                    rest = threshold - file_size[load_buffer_ptr]
                                    wait_model.acquire()
                                time.sleep(0.1)
                        with block_lock:
                            block_nums.value -= 1
                    if wait_model._value == 0:
                        load_buffer.put(load_buffer_ptr)
                        load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                else:
                    load_buffer.put(load_buffer_ptr)
                    load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                    if len(model_buffer) < buffer_size and rest - psutil.virtual_memory().used > (100 << 20):
                        load_buffer.put(load_buffer_ptr)
                        load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                wait_model.acquire()
                input = model_buffer[layer](input)
                del_buffer.put(del_buffer_ptr)
                del_buffer_ptr = (del_buffer_ptr + 1) % layers_num
    print('resolution --', resolution, ':', time.time() - start)
    max_resolution.value = 608
    print('-------------------------------------------------------------------')
    
def trigger(pipe, pipe_lock):
    global enough
    while True:
        if pipe.recv():
            with pipe_lock:
                enough = False
            pipe.send(True)

def worker_listener(queue, enough_lock):
    threads = 1
    global thread_base, running, enough
    # print(thread_base)
    while True:
        temp = queue.get()
        if running == False:                #结束，防止无效接收
            queue.put(temp + 1)
            time.sleep(0.1)
            break
        if temp == 200:                     #外部开辟了新的推理流水
            if thread_base == 1:
                queue.put(temp)
                time.sleep(0.1)
            else:
                thread_base -= 1
                torch.set_num_threads(thread_base)

        if temp > 16:
            pass
        elif temp > 10:
            if enough:
                queue.put(temp)
                time.sleep(0.1)
            else:
                print(temp)
                with enough_lock:
                    enough = True
        elif temp > 0:
            thread_base += temp
            torch.set_num_threads(thread_base)
        else:
            with enough_lock:
                enough = True
            threads += 1
            thread_base -= 1
            torch.set_num_threads(thread_base)

def Inference2(resolution, threshold, exceed_lock, block_lock, block_nums, free_nums, main_pid, mem_size, file_size, base, add_lock, child, wait_queue):
    start = time.time()
    sef_pid = os.getpid()
    global enough, thread_base, running
    running = True
    thread_base = base
    enough= False
    main_process = psutil.Process(main_pid)
    layers_num, batch_size, task_num = 57, 1, 10
    model_buffer = {}
    buffer_size = 3
    load_buffer, del_buffer = queue.Queue(), queue.Queue()
    wait_model, enough_lock = threading.Semaphore(0), threading.Lock()
    loading = threading.Thread(target=load_model, args=(model_buffer, load_buffer, wait_model))
    deleting = threading.Thread(target=delete_model, args=(model_buffer, del_buffer))
    # listner = threading.Thread(target=worker_listener, args=(worker_queue, enough_lock, add_lock))
    # listner.daemon = True
    loading.daemon = True
    deleting.daemon = True
    loading.start()
    deleting.start()
    # listner.start()
    load_buffer_ptr = 0
    del_buffer_ptr = 0
    for i in range(buffer_size):
        load_buffer.put(load_buffer_ptr)
        load_buffer_ptr += 1
    if resolution == 608:
        high = mem_size[6] + (2 << 20)
    else:
        high = (19 + 24) << 20
    torch.set_num_threads(thread_base)
    memory = 0
    for p in main_process.children():
        memory += p.memory_info().rss
    print('init:', psutil.Process(sef_pid).memory_info().rss >> 20)
    listener_time = 0
    for i in range(task_num):
        input = torch.randn(batch_size, 3, resolution, resolution)
        with torch.no_grad():
            layer = 0
            while layer < layers_num:
                    with add_lock:
                        if free_nums.value > 0:
                            thread_base += free_nums.value
                            free_nums.value = 0
                            torch.set_num_threads(thread_base)
                    s = time.time()
                    try:
                        memory = 0
                        for p in main_process.children():
                            memory += p.memory_info().rss
                    except:
                        memory = 0
                        for p in main_process.children():
                            memory += p.memory_info().rss
                        continue
                    listener_time += time.time() - s
                    cha = threshold - mem_size[layer] * batch_size - memory #mem_informer.value
                    if wait_model._value == 0:
                        cha -= file_size[load_buffer_ptr]
                    if cha < high and (layer < 7 or (layer >= 50 and layer < 55)):
                        with block_lock:
                            block_nums.value += thread_base
                        with exceed_lock:
                            with block_lock:
                                block_nums.value -= thread_base
                            while layer < 7 or (layer >= 50 and layer < 54):
                                with block_lock:
                                    torch.set_num_threads(thread_base + block_nums.value)
                                if wait_model._value == 0:
                                    load_buffer.put(load_buffer_ptr)
                                    load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                                wait_model.acquire()                        
                                input = model_buffer[layer](input)
                                del_buffer.put(del_buffer_ptr)
                                del_buffer_ptr = (del_buffer_ptr + 1) % layers_num
                                layer += 1
                            layer -= 1
                        torch.set_num_threads(thread_base)
                    elif cha <= 0:
                        if wait_model._value == 0:
                            load_buffer.put(load_buffer_ptr)
                            load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                        wait_model.acquire()                        
                        input = model_buffer[layer](input)
                        del_buffer.put(del_buffer_ptr)
                        del_buffer_ptr = (del_buffer_ptr + 1) % layers_num
                    else:
                        load_buffer.put(load_buffer_ptr)
                        load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                        wait_model.acquire()                        
                        input = model_buffer[layer](input)
                        del_buffer.put(del_buffer_ptr)
                        del_buffer_ptr = (del_buffer_ptr + 1) % layers_num
                        if len(model_buffer) < buffer_size and cha > file_size[51]:
                            load_buffer.put(load_buffer_ptr)
                            load_buffer_ptr = (load_buffer_ptr + 1) % layers_num
                    layer += 1
    whole_time = time.time() - start
    running = False
    if wait_queue.empty():
        with add_lock:
            free_nums.value += thread_base
    else:
        with add_lock:
            child.send(resolution)
            free_nums.value += thread_base - 1
    print('resnet resolution --', resolution, ':', time.time() - start, listener_time, whole_time - listener_time, thread_base)
    print('-------------------------------------------------------------------')
    
def listen_over(coon, worker_queue, is_waiting):
    global set_base
    set_base = 0
    while True:
        temp = coon.recv()
        if is_waiting > 0:
            set_base = temp
            time.sleep(1)
        else:
            time.sleep(1)
            worker_queue.put(temp)
            

def mem_info():          ## 内存监控函数
    global max_mem, real_memory
    process = psutil.Process(os.getpid())
    while True:
        try:
            temp = 0
            for p in process.children():
                temp += p.memory_info().rss
        except:
            temp= 0
            for p in process.children():
                temp += p.memory_info().rss
            continue
        if max_mem < temp:
            max_mem = temp

        time.sleep(0.1)
if __name__ == '__main__':
    memory_records = 0
    mem_size = {}
    mem_size[608] = resnet_mem(resolution=608)
    mem_size[224] = resnet_mem(resolution=224)
    file_size = get_filesize('resnet152')
    is_waiting = mp.Value('i', 0)
    wait_queue = mp.Queue()
    max_mem, max_cpu, cpu, threshold, real_memory = 0, 0, 0, 0, 0
    thread_used = mp.Value('i', 0)
    torch.set_num_threads(1)
    worker_queue = mp.Queue()
    mem_listener = threading.Thread(target=mem_info)
    mem_listener.daemon = True
    mem_listener.start()
    process = []
    gc.collect()
    time.sleep(1)
    start = time.time()
    child, parent = mp.Pipe()
    pipe_608, pipe_224 = 2, 0

    for i in range(pipe_224):
        process.append(mp.Process(target=no_block, args=(224, child)))
    for i in range(pipe_608):
        process.append(mp.Process(target=no_block, args=(608, child)))
    
    for i in process:
        i.demon = True
    for i in process:
        i.start()
    # if parent.recv():
    #     p = mp.Process(target=no_block, args=(608, child))
    #     p.daemon = True
    #     p.start()
    #     process.append(p)
    for i in process:
        i.join()
    print(time.time() - start, max_mem >> 20, max_cpu / 100, cpu / 5)
    for i in range(2):
        parent.recv()
    threshold = 1400
    print('threshold: ', threshold)
    max_mem, max_cpu, cpu = 0, 0, 0
    # exit()
    block_lock, exceed_block, block_nums, cpu_used = mp.Lock, mp.Lock, mp.Value('i', 0), mp.Value('i', 0)
    main_process = psutil.Process(os.getpid())

    
    process = []
    gc.collect()
    time.sleep(1)
    block_lock, exceed_block = mp.Lock(), mp.Lock()
    block_nums, free_nums, add_lock = mp.Value('i', 0), mp.Value('i', 0), mp.Lock()
    start = time.time()
    for i in range(pipe_224):
        process.append(mp.Process(target=Inference2, args=(224, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[608], file_size, 1, add_lock, child, wait_queue)))
    for i in range(pipe_608):
        process.append(mp.Process(target=Inference2, args=(608, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[608], file_size, 1, add_lock, child, wait_queue)))
    free_nums.value = 3
    # wait_queue.put(608)
    for i in process:
        i.demon = True
    for i in process:
        i.start()
    while(wait_queue.empty() == False):
        if len(main_process.children()) < 5 and threshold - real_memory > (500 << 20):
                time.sleep(1)
                temp = wait_queue.get()
                p = mp.Process(target=Inference2, args=(temp, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[temp], file_size, 1, add_lock, child, wait_queue))
                p.start()
                process.append(p)
                with block_lock:
                    is_waiting.value = wait_queue.qsize()
        else:
                parent.recv()
                temp = wait_queue.get()
                p = mp.Process(target=Inference2, args=(temp, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[temp], file_size, 1, add_lock, child, wait_queue))
                p.start()
                process.append(p)
                with block_lock:
                    is_waiting.value = wait_queue.qsize()
    time.sleep(0.5)
    for i in process:
        i.join()
    print(time.time() - start, max_mem >> 20, max_cpu / 100, cpu / 5)


    


