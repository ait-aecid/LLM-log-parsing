import logging
import queue
import threading
import time
from online_run import onlineWork

end_of_data = object()
input_tokens = []
output_tokens = []
free_times = []
q_size = []
new_logs_second = []

def read_and_enqueue(file_path, data_queue, cond, interval=3, lines_per_read=500):
    num = 0
    lines = []
    with open(file_path, 'r') as file:
        while True:
            lines += [line.strip() for _, line in zip(range(lines_per_read), file)]
            print("Putting logs in queue:", num)
            print("line",  lines)
            if not lines or num >= 600:
                break
            with cond:
                while len(lines) >=15000:
                    data_queue.put(lines[:15000])
                    lines = lines[15000:]
                print("size number:", data_queue.qsize()*15000)
                q_size.append(data_queue.qsize()*15000)
                cond.notify()
            #time.sleep(interval)
            num += 1
        with cond:
            data_queue.put(end_of_data)
            cond.notify()


def process_data(data_batch, oracle):
    print("Processing batch of data:", len(data_batch))
    input_token,output_token,free_time,new_logs = onlineWork(data_batch, oracle)
    input_tokens.append(input_token)
    output_tokens.append(output_token)
    free_times.append(free_time)
    new_logs_second.append(new_logs)
    print("free time:", free_time)
    print("new_logs_second", new_logs)


def consumer(data_queue, cond, oracle, prod_thread, interval):
    while True:
        with cond:
            while data_queue.empty():
                if not prod_thread.is_alive():
                    return
                cond.wait(interval)
            data_batch = data_queue.get()
        if data_batch == end_of_data:
            return
        process_data(data_batch, oracle)
        print("processing time:")


def main(file_path, interval=1, lines_per_read=10):
    q = queue.Queue()
    cond = threading.Condition()
    prod_thread = threading.Thread(target=read_and_enqueue, args=(file_path, q, cond, interval, lines_per_read))
    cons_thread = threading.Thread(target=consumer, args=(q, cond, oracle, prod_thread, interval))
    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()

    print("Data processing complete.")


file_path = '/home/beckv/projects/LLMs_for_log_parsing/data/2k/Apache/Apache_2k.log' # full log path
oracle = '' # dataset name
model_path = "" # embdding model path

if __name__ == "__main__":
    main(file_path)
    # total_evaluate(oracle)
    print(input_tokens)
    print(sum(input_tokens))
    print(output_tokens)
    print(sum(output_tokens))
    print(len(input_tokens))
    print(output_tokens.count(0))

    print(q_size)
    print(sum(q_size))
    print(sum(q_size)/len(q_size))
    print(free_times)
    print(sum(free_times))
    print(new_logs_second)
    print(sum(new_logs_second))
    # logging.info(f'{user} logged in!')
    # logging.info(f'Reading data from {input_file_path}')
    # logging.error(f'Failed writing data to {output_file_path}')
    # logging.warning(f'System is running out of memory!')

