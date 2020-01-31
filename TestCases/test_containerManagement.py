import pytest
import requests
import json

headers = { "key": "Content-Type","value": "application/json"}
                                                                            

def test_getClusterIds():
    res = requests.post("http://sredevvm.southcentralus.cloudapp.azure.com:9026/managecontainers",headers = headers, data = '{"type":"getClusterIds"}')
    res_code = res.status_code
    res_json = res.json()
    assert res_code == 200
    assert res_json['Status'] == 'Success'

def test_listContainers():
    res = requests.post("http://sredevvm.southcentralus.cloudapp.azure.com:9026/managecontainers",headers = headers, data='{"type":"listContainers","optype":"cluster","id":"1"}' )
    res_code = res.status_code
    res_json = res.json()
    assert res_code == 200
    assert res_json['Status'] == 'Success'

def test_stopContainers():

    data = '{"type":"managecontainers","data":{"exectype":"serial","action":"stop","contname":["PROJ-CORPSOAPMg_18-011281-R09_001_00_149_D6"],"vmip":"172.16.0.164"}}'
    res = requests.post("http://sredevvm.southcentralus.cloudapp.azure.com:9026/managecontainers",headers = headers, data=data )
    res_code = res.status_code
    res_json = res.json()
    print(res_json)
    assert res_code == 300
    assert res_json['Status'] == 'Success'

def test_startContainers():
    data = '{"type":"managecontainers","data":{"exectype":"serial","action":"start","contname":["PROJ-CORPSOAPMg_18-011281-R09_001_00_149_D6"],"vmip":"172.16.0.164"}}'
    res = requests.post("http://sredevvm.southcentralus.cloudapp.azure.com:9026/managecontainers",headers = headers, data=data )
    res_code = res.status_code
    res_json = res.json()
    print(res_json)
    assert res_code == 200
    assert res_json['Status'] == 'Success'



