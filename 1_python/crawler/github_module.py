import requests

# github_module.py


proxies = {
        #          [协议]://  [地址]  :[端口]
        "http":  "socks5h://127.0.0.1:7890",  # 再例如  "http":  "http://127.0.0.1:7890",
        "https": "socks5h://127.0.0.1:7890",  # 再例如  "https": "http://127.0.0.1:7890",
    }

#  https://docs.github.com/zh/rest/search?apiVersion=2022-11-28

def get_top_java_repositories(page, per_page):
    """
    详细描述

    :param page: 获取第几页
    :param per_page: 当前页下获取几条记录
    """

    url = 'https://api.github.com/search/repositories'
    # 使用 Authorization指使用github的有认证方式
    headers = {
        'Authorization': 'Bearer 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',
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

if __name__ == "__main__":
    repositories = get_top_java_repositories(1, 5)
    print(repositories)
    print('\n\n')
    for repos in repositories:
        print(repos['clone_url'])

# clone_url
# https://github.com/elastic/elasticsearch.git
# https://github.com/TheAlgorithms/Java.git
# https://github.com/krahets/hello-algo.git
# https://github.com/zxing/zxing.git
# https://github.com/alibaba/nacos.git