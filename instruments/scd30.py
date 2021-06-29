import requests
from requests.auth import HTTPBasicAuth
import json


class SCD30:

    def __init__(self, uri: str, username: str, password: str):
        self.__uri = uri
        self.__username = username
        self.__password = password

    def read_env(self) -> list:
        url = self.__uri + '/env'
        resp = requests.get(url=url)
        data = resp.json()
        return data


