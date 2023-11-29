import time
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback

# import clone_repo_module as crm

from clone_repo_module import clone_repo

# thread_pool_executor__module.py

# def thread_pool_executor():
#     with ThreadPoolExecutor(max_workers=10) as executor:  # 这里使用了最多10个线程的线程池
#             results = []
#             futures = []
#             for i in range(0,10):
#
#                 future = executor.submit(run, 'param...', i)
#                 futures.append(future)
#
#             for future in futures:
#                 results.append(future.result())
#             return results
#
# def run(param1, param2):
#     var = f'param1 = {param1}, param2 = {param2} ,, Running in thread: {threading.current_thread().name}'
#     print(var)
#     # time.sleep(1)
#     return var
#
# if __name__ == '__main__':
#     results = thread_pool_executor()
#     for res in results:
#         print(f'线程返回 = {res}')

def thread_pool_executor(repo_urls):
    with ThreadPoolExecutor(max_workers=10) as executor:  # 这里使用了最多10个线程的线程池
            results = []
            futures = []
            for repo_url in repo_urls:
                future = executor.submit(run, repo_url)
                futures.append(future)

            for future in futures:
                results.append(future.result())
            return results

def run(repo_url):
    local_repo_path = None
    try:
        print(f'Running in thread: {threading.current_thread().name}，仓库：{repo_url}，开始下载')
        # local_repo_path = crm.clone_repo(repo_url)
        local_repo_path = clone_repo(repo_url)
        print(f'Running in thread: {threading.current_thread().name}，仓库：{repo_url}，下载完成')
    except Exception as e:
        # 捕获其他所有异常
        # print(f"发生了其他异常: {e}")
        stack_trace = traceback.format_exc(limit=1)
        local_repo_path = stack_trace
    return local_repo_path

if __name__ == '__main__':
    repo_urls = ['https://github.com/nhjclxc/java-callgraph-spoon.git', 'https://github.com/nhjclxc/netty-test.git',
                 'https://github.com/nhjclxc/chatgpt4nhjclxc.git',
                 'https://github.com/nhjclxc/design-model.git',
                 'https://github.com/nhjclxc/repository_test.git']  # , 'https://github.com/zxing/zxing.git']

    results = thread_pool_executor(repo_urls)
    for res in results:
        print(f'线程返回 = {res}')




# class CloneRepoThreadPool:
#     """
#         定义线程池
#     """
#     def __init__(self):
#         self.executor = ThreadPoolExecutor(max_workers=10)
#
#     def thread_pool_executor(self, repo_urls):
#         results = []
#         for repo_url in repo_urls:
#             future = self.executor.submit(self.run, repo_url)
#             results.append(future.result())
#         return results
#
#
#     def run(self, repo_url):
#         """
#             线程池任务
#         :param repo_url: 仓库地址
#         :return: 返回下载完成的本地仓库路径（相对路径）
#         """
#         local_repo_path = None
#         try:
#             print(f'Running in thread: {threading.current_thread().name}')
#             local_repo_path = crm.clone_repo(repo_url)
#             print(f'仓库：{repo_url}，下载完成')
#         except Exception as e:
#             # 捕获其他所有异常
#             # print(f"发生了其他异常: {e}")
#             stack_trace = traceback.format_exc(limit=1)
#             local_repo_path = stack_trace
#         return local_repo_path


# repo_urls = ['https://github.com/nhjclxc/java-callgraph-spoon.git', 'https://github.com/nhjclxc/netty-test.git', 'https://github.com/nhjclxc/chatgpt4nhjclxc.git',
#              'https://github.com/nhjclxc/design-model.git','https://github.com/nhjclxc/repository_test.git'] #, 'https://github.com/zxing/zxing.git']
# thread_pool = CloneRepoThreadPool()
# results = thread_pool.thread_pool_executor(repo_urls)
# for res in results:
#     print(f'线程返回 = {res}')

