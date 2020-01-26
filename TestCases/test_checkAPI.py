import pytest
import requests
import json

res = requests.get('https://jsonplaceholder.typicode.com/todos/1')
def test_API1():
    res = json.loads(requests.get('https://jsonplaceholder.typicode.com/todos/1').content)
    assert res["userId"]==1

def test_API2():
    res = json.loads(requests.get('https://jsonplaceholder.typicode.com/users').content)
    assert len(res[0].keys()) == 8
