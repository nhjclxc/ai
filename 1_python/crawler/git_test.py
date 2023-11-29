# import requests
# from git import Repo # pip install Repo
# import os
# import shutil
# from concurrent.futures import ThreadPoolExecutor
#
#
# def clone_java_repositories(num_repos):
#     page = 1
#     per_page = 100  # 每页数量
#     cloned_repos = 0
#
#     with ThreadPoolExecutor(max_workers=10) as executor:  # 这里使用了最多10个线程的线程池
#         while cloned_repos < num_repos:
#             repositories = get_top_java_repositories(page, per_page)
#
#             if not repositories:  # 如果获取到的仓库列表为空，结束循环
#                 break
#
#             futures = []
#             for repo in repositories:
#                 if cloned_repos >= num_repos:  # 达到目标数量，结束循环
#                     break
#
#                 future = executor.submit(clone_and_filter_java_files, repo['clone_url'])
#                 futures.append(future)
#                 cloned_repos += 1
#
#             for future in futures:
#                 future.result()
#
#             page += 1
#
#
# def get_top_java_repositories(page, per_page):
#     headers = {
#         'Authorization': 'Bearer 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'  # 替换为你的GitHub Token
#     }
#     params = {
#         'q': 'language:java',
#         'sort': 'stars',
#         'order': 'desc',
#         'page': page,
#         'per_page': per_page
#     }
#     url = 'https://api.github.com/search/repositories'
#     response = requests.get(url, params=params, headers=headers)
#     if response.status_code == 200:
#         repositories = response.json()['items']
#         return repositories
#     else:
#         print('Failed to fetch repositories')
#         return []
#
#
# def clone_and_filter_java_files(repo_url):
#     local_repo_path = 'E:\nbu\ai\1_python\git_path' #'/path/to/local/repo'
#     repo = Repo.clone_from(repo_url, local_repo_path)
#
#     java_files = [file for file in os.listdir(local_repo_path) if file.endswith('.java')]
#     for file in os.listdir(local_repo_path):
#         file_path = os.path.join(local_repo_path, file)
#         if file not in java_files:
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#
#
# if __name__ == "__main__":
#     num_repos_to_download = 2 # 2000
#     clone_java_repositories(num_repos_to_download)