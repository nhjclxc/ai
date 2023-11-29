import os
import shutil
import stat

from git.repo import Repo

# clone_repo_module.py

local_repo_path_prefix = 'git_repo'
''' 本地仓库路径前缀 '''

def clone_repo(remote_url):
    """
        克隆仓库到本地
    :param repo_url: 远程仓库地址
    :return: 下载后的本地仓库地址
    """
    # 从 URL 中提取仓库名称
    repo_name = os.path.splitext(os.path.basename(remote_url))[0]
    local_repo_path = os.path.join(local_repo_path_prefix, repo_name)  #拼接本地下载路径前缀 'git_repo'
    # print(f'local_repo_path = {local_repo_path}')
    repo = Repo.clone_from(remote_url, local_repo_path)
    # print(f'repo = {repo}')

    # 由于在克隆时，无法直接过滤文件，所以只能在过滤之后采用曲线的方法，在本地清除不需要的文件
    filter_file(local_repo_path)

    return local_repo_path


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


if __name__ == "__main__":
    remote_url = 'https://github.com/nhjclxc/java-callgraph-spoon.git'
    local_repo_path = clone_repo(remote_url)
    print(f'local_repo_path = {local_repo_path}')

# https://github.com/elastic/elasticsearch.git
# https://github.com/TheAlgorithms/Java.git
# https://github.com/krahets/hello-algo.git
# https://github.com/zxing/zxing.git
# https://github.com/alibaba/nacos.git
