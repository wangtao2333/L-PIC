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
        model_buffer[idx] = torch.load('Regnet/part' + str(idx) + '.bin').eval()
        wait_model.release()
            
## 删除模型
def delete_model(model_buffer, del_buffer):
    while True:
        del model_buffer[del_buffer.get()]

def resnet_mem(resolution = 608):
    dirname = os.path.abspath (os.path.dirname (__file__))
    out_shape = []
    shape0 = resolution * resolution * 3
    model = torch.load(dirname + '/Regnet/part0.bin')
    channel = model.out_channels
    resolution = (resolution - model.kernel_size[0] + (model.padding[0] << 1)) // model.stride[0] + 1
    shape0 = resolution * resolution * channel
    out_shape = [shape0 << 3, shape0 << 3, shape0 << 3]
    for part in range(3, 20):
        model = torch.load(dirname + '/Regnet/part' + str(part) + '.bin')
        if model.proj == None:
            out_shape.append(shape0 * 3 << 2)
        else:
            temp = shape0
            channel = model.proj[0].out_channels
            temp2 = channel * resolution * resolution << 1
            resolution = (resolution - model.proj[0].kernel_size[0]) // model.proj[0].stride[0] + 1
            shape0 = resolution * resolution * channel
            out_shape.append(temp + temp2 + shape0 << 2)
    out_shape.append(channel + shape0 << 2)
    out_shape.append(channel << 3)
    model = torch.load(dirname + '/Regnet/part22.bin')
    out_shape.append(model.in_features + model.out_features << 2)
    return out_shape
    

def get_filesize(model):
    layer_nums = 0
    if model == 'Regnet':
        layer_nums = 23
    if model == 'convnext_base':
        layer_nums = 42
    if model == 'resnet152':
        layer_nums = 57
    # path = model + '_layers'
    path = model
    file_size = []
    for layer in range(layer_nums):
        file_size.append(os.path.getsize(path + '/part' + str(layer) + '.bin'))
    return file_size
    
def no_block(resolution, child):
    start = time.time()
    layers_num, batch_size, task_num = 23, 1, 10
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
    
def trigger(pipe, pipe_lock):
    global enough
    while True:
        if pipe.recv():
            with pipe_lock:
                enough = False
            pipe.send(True)

def worker_listener(worker_queue, enough_lock, add_lock):
    threads = 1
    global thread_base, running, enough
    # print(thread_base)
    while running:
        # with add_lock:
        temp = worker_queue.get()
        # temp = parent_add.recv()  #.get()
        print('gett')
        if running == False:                #结束，防止无效接收
            queue.send(temp + 1)
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

def Inference2(id, resolution, threshold, exceed_lock, block_lock, block_nums, free_nums, main_pid, mem_size, file_size, base, add_lock, child, wait_queue):
    start = time.time()
    sef_pid = os.getpid()
    global enough, thread_base, running
    running = True
    thread_base = base
    enough= False
    main_process = psutil.Process(main_pid)
    layers_num, batch_size, task_num = 23, 1, 10
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
        high = mem_size[3] + (2 << 20)
    else:
        high = file_size[19] + mem_size[19]
    torch.set_num_threads(thread_base)
    temp = False
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
                    if cha <= high and ((layer < 6 and layer > 0) or (layer < 11 and layer > 7) or (layer > 17 and layer < 20)):
                        with block_lock:
                            block_nums.value += thread_base
                        with exceed_lock:
                            with block_lock:
                                block_nums.value -= thread_base
                            while (layer < 6 or (layer < 11 and layer > 7) or (layer > 17 and layer < 20)):
                                with block_lock:
                                    torch.set_num_threads(block_nums.value + thread_base)
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
                    # else:
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
                        if len(model_buffer) < buffer_size and cha > (file_size[19] << 20):
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
        # if is_waiting.value > 0:
        #     is_waiting.value -= 1
    print('resolution --', resolution, ':', id, time.time() - start, listener_time, whole_time - listener_time, thread_base)
    print('-------------------------------------------------------------------')
    
def listen_over(coon, worker_queue, wait_queue, ready):
    global set_base
    set_base = 0
    while True:
        temp = coon.recv()
        # print('get')
        # print(is_waiting.value)
        if wait_queue.empty() == False:
            ready.put(True)
            # print('sent')
            # for i in range(temp - 1):
                # worker_queue.put(1)
                # time.sleep(2)
                # parent_add.send(1) #
        # else:
        #     worker_queue.put(1)
            # time.sleep(2)
            # parent_add.send(temp)
            # for i in range(temp):
                # time.sleep(0.35)
                # parent_add.send(1) #worker_queue.put(1)
            

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
        real_memory = temp
        if max_mem < temp:
            max_mem = temp

        time.sleep(0.1)
if __name__ == '__main__':
    memory_records = 0
    mem_size = {}
    mem_size[608] = resnet_mem(resolution=608)
    mem_size[224] = resnet_mem(resolution=224)
    file_size = get_filesize('Regnet')
    # for i in range(57):
    #     print(mem_size[608][i] + file_size[i] >> 20)
    # exit()
    free_nums = mp.Value('i', 0)
    wait_queue = mp.Queue()
    max_mem, max_cpu, cpu, threshold = 0, 0, 0, 0
    real_memory = 0
    thread_used = mp.Value('i', 0)
    torch.set_num_threads(1)
    # worker_queue = mp.Queue()
    mem_listener = threading.Thread(target=mem_info)
    mem_listener.daemon = True
    mem_listener.start()
    process = []
    child, parent = mp.Pipe()
    gc.collect()
    time.sleep(1)
    start = time.time()
    pipe_608, pipe_224 = 4, 0

    for i in range(pipe_608 - 2):
        process.append(mp.Process(target=no_block, args=(608, child)))
    for i in range(pipe_224):
        process.append(mp.Process(target=no_block, args=(224, child)))
    for i in process:
        i.demon = True
    for i in process:
        i.start()
    if parent.recv():
        p = mp.Process(target=no_block, args=(608, child))
        p.daemon = True
        p.start()
        process.append(p)
    # if parent.recv():
    #     p = mp.Process(target=no_block, args=(608, child))
    #     p.daemon = True
    #     p.start()
    #     process.append(p)
    
    if parent.recv():
        p = mp.Process(target=no_block, args=(608, child))
        p.daemon = True
        p.start()
        process.append(p)
    for i in process:
        i.join()
    
    print(time.time() - start, max_mem >> 20, max_cpu / 100, cpu / 5)
    for i in range(2):
        parent.recv()
    threshold = 1400
    is_waiting = mp.Value('i', 0)
    print('threshold: ', threshold)
    max_mem, max_cpu, cpu = 0, 0, 0
    # exit()
    parent_add, child_add = mp.Pipe()
    parent_add.send(1)
    child_add.recv()
    block_lock, exceed_block, block_nums, cpu_used = mp.Lock(), mp.Lock(), mp.Value('i', 0), mp.Value('i', 0)
    main_process = psutil.Process(os.getpid())
    ready = mp.Queue()
    add_lock = mp.Lock()
    # listening_over = mp.Process(target=listen_over, args=(parent , worker_queue, wait_queue, ready))
    # listening_over.daemon = True
    # listening_over.start()
    process = []
    gc.collect()
    time.sleep(1)
    num = 0
    # block_lock, exceed_block = mp.Lock(), mp.Lock()
    block_nums = mp.Value('i', 0)
    start = time.time()
    for i in range(pipe_224):
        process.append(mp.Process(target=Inference2, args=(num, 224, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[224], file_size, 1, add_lock, child, wait_queue)))
        num += 1
    # process.append(mp.Process(target=Inference2, args=(224, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[224], file_size, 1, add_lock, child)))
    for i in range(pipe_608 - 2):
        process.append(mp.Process(target=Inference2, args=(num, 608, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[608], file_size, 1, add_lock, child, wait_queue)))
        num += 1
    free_nums.value = 3
    wait_queue.put(608)
    wait_queue.put(608)
    # wait_queue.put(608)
    is_waiting.value = wait_queue.qsize()
    # print(is_waiting.value)
    for i in process:
        i.demon = True
    for i in process:
        i.start()
    # time.sleep(1)
    # parent_add.send(1)
    # num = 0
    while(wait_queue.empty() == False):
        if len(main_process.children()) < 5 and threshold - real_memory > (500 << 20):
            resolution = wait_queue.get()
            parent_add.send(200)
            # time.sleep(1)
            p = mp.Process(target=Inference2, args=(num, resolution, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[resolution], file_size, 1, add_lock, child, wait_queue))
            p.start()
            num += 1
            process.append(p)
            with block_lock:
                is_waiting.value = wait_queue.qsize()
        if parent.recv() == 608:
            # print('built')
            resolution = wait_queue.get()
            gc.collect()
            p = mp.Process(target=Inference2, args=(num, resolution, threshold << 20, exceed_block, block_lock, block_nums, free_nums, os.getpid(), mem_size[resolution], file_size, 1, add_lock, child, wait_queue))
            p.daemon = True
            p.start()
            num += 1
            process.append(p)
            with block_lock:
                is_waiting.value = wait_queue.qsize()
            # print('yes')
            time.sleep(0.3)
            # worker_queue.put(1)
            # parent_add.send(1)

    time.sleep(0.5)
    # print(worker_queue.get())
    for i in process:
        i.join()
    print(time.time() - start, max_mem >> 20, max_cpu / 100, cpu / 5)


    

