import threading
import traceback
import requests
import os
import shutil
import stat

from concurrent.futures import ThreadPoolExecutor
from git.repo import Repo
# https://www.cnblogs.com/kai-/p/12705522.html

# from github_module import get_top_java_repositories
# from clone_repo_module import clone_repo
# import thread_pool_executor__module as tpem

# 获取仓库克隆地址
proxies = {
        #          [协议]://  [地址]  :[端口]
        "http":  "socks5h://127.0.0.1:7890",  # 再例如  "http":  "http://127.0.0.1:7890",
        "https": "socks5h://127.0.0.1:7890",  # 再例如  "https": "http://127.0.0.1:7890",
    }
''' 防止出现代理问题 '''

def get_top_java_repositories(page, per_page):
    """
    获取java语言仓库start最多的
        #  https://docs.github.com/zh/rest/search?apiVersion=2022-11-28
    :param page: 获取第几页
    :param per_page: 当前页下获取几条记录
    """

    url = 'https://api.github.com/search/repositories'
    # 使用 Authorization指使用github的有认证方式
    headers = {
        'Authorization': 'Bearer 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', # YOUR_GITHUB_TOKEN
        'Accept': 'application/vnd.github.v3+json'
    }
    params = {
        'q': 'language:java',
        'sort': 'stars',
        'order': 'desc',
        'page': page,
        'per_page': per_page
    }
    response = requests.get(url, params=params, headers=headers, proxies= proxies)

    if response.status_code == 200:
        repositories = response.json()['items']
        return repositories
        # for repo in repositories:
        #     print(f"Repository: {repo['name']}, Stars: {repo['stargazers_count']}")
    else:
        print('Failed to fetch repositories')
        return []


# 克隆仓库
local_repo_path_prefix = '..\data_set\src'
''' 本地仓库路径前缀 '''

def clone_repo(remote_url):
    """
        克隆仓库到本地
    :param repo_url: 远程仓库地址
    :return: 下载后的本地仓库地址
    """

    local_repo_path = None
    try:
        print(f'Running in thread: {threading.current_thread().name}，仓库开始下载：{remote_url}，')
        # local_repo_path = crm.clone_repo(repo_url)

        # 从 URL 中提取仓库名称
        repo_name = os.path.splitext(os.path.basename(remote_url))[0]
        local_repo_path = os.path.join(local_repo_path_prefix, repo_name)  # 拼接本地下载路径前缀 'git_repo'
        # print(f'local_repo_path = {local_repo_path}')
        repo = Repo.clone_from(remote_url, local_repo_path)
        # print(f'repo = {repo}')

        # 由于在克隆时，无法直接过滤文件，所以只能在过滤之后采用曲线的方法，在本地清除不需要的文件
        filter_file(local_repo_path)

        print(f'Running in thread: {threading.current_thread().name}，仓库下载完成：{remote_url}，')
        return remote_url
    except Exception as e:
        # 捕获其他所有异常
        # print(f"发生了其他异常: {e}")
        stack_trace = traceback.format_exc(limit=1)
        local_repo_path = stack_trace
        return None
    # return local_repo_path

def filter_file(local_repo_path, file_suffix = '.java'):
    """
        过滤路径下的文件
    :param local_repo_path: 路径
    :param save_file_suffix: 要保留的文件后缀
    :return:
    """
    java_files = [file for file in os.listdir(local_repo_path) if file.endswith(file_suffix)]
    for file in os.listdir(local_repo_path):
        file_path = os.path.join(local_repo_path, file)
        if file not in java_files:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                if '.git' in file_path:  #由于.git文件下的.ind文件存在访问权限问题，所以直接删除.git文件夹
                    clear_folder(file_path)
                filter_file(file_path)  # 递归清理子目录

                delete_empty_folder(file_path)

def delete_empty_folder(folder_path):
    """
        判断当前文件夹是否为空，为空则删除文件夹
    :param folder_path: 当前要清空的文件夹
    :return:
    """
    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 检查文件夹是否为空
        if not os.listdir(folder_path):
            # 文件夹为空，删除文件夹
            os.rmdir(folder_path)

def clear_folder(path):
    """
    clear specified folder
    https://my.oschina.net/hechunc/blog/3078597
    :param path: the path where need to clear.
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path, onerror=readonly_handler)
    # time.sleep(1)
    os.mkdir(path)

def readonly_handler(func, path, execinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def write_remote_repo_to_file(results):
    try:
        with open(os.path.join(local_repo_path_prefix, "git_repo_urls.txt"), "a+", encoding='utf-8') as f:
            for res in results:
                f.write(res + '\n')  # 自带文件关闭功能，不需要再写f.close()
    except Exception as e:
        print(f"write_remote_repo_to_file 发生了其他异常: {e}")
        stack_trace = traceback.format_exc()

def clone_java_repositories(num_repos):
    page = 1
    per_page = 100  # 每页数量，最大值为100
    cloned_repos = 0 #已克隆仓库数量
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:  # 这里使用了最多10个线程的线程池
        while cloned_repos < num_repos:
            repositories = get_top_java_repositories(page, per_page)

            if not repositories:  # 如果获取到的仓库列表为空，结束循环
                break

            futures = []
            for repo in repositories:
                if cloned_repos >= num_repos:  # 达到目标数量，结束循环
                    break
                repo_url = repo['clone_url']
                if repo_url in results:
                    continue #该仓库已存在，则跳过这个仓库

                future = executor.submit(clone_repo, repo_url)
                futures.append(future)
                cloned_repos += 1

            for future in futures:
                results.append(future.result())

            page += 1

    # 将所有git仓库路径写入文件
    write_remote_repo_to_file(results)

    return results



if __name__ == "__main__":
    num_repos_to_download = 1
    results = clone_java_repositories(num_repos_to_download)

    pass
