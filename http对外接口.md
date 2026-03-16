# HTTP 对外接口

## 1. 请求采集控制

- 方法: `POST`
- URL: `http://ip:8088/raman/jy/capture`
- Content-Type: `application/json`

请求 Body:

```json
{
  "req_id": "123",
  "capture": {
    "explore_time": 1,
    "integer": 5,
    "laser": 20,
    "center_wave": 855.25,
    ...
  }
}
```

返回:

```json
{
  "code": 0,
  "msg": ""
}
```

## 2. 回传数据接口

- 方法: `POST`
- URL: `http://ip:port/raman/jy/callback`

请求 Body:

```json
{
  "req_id": "123",
  "data": {
    "x": [123, 123, 123.0],
    "y": [12312, 123, 123]
  }
}
```

> `x` — 拉曼位移
> `y` — 强度

返回:

```json
{
  "code": 0,
  "msg": ""
}
```
