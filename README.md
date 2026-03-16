| 原接口文档                     | 代码实现                    | 备注                    |
| ------------------------- | ----------------------- | --------------------- |
| `POST /raman/jy/capture`  | `self.capture_endpoint` | 发送采集指令                |
| `POST /raman/jy/callback` | 启动`HTTPServer`接收        | 仪器主动推送数据              |
| `req_id`                  | `uuid.uuid4()`生成        | 唯一标识关联请求和回调           |
| `x` (拉曼位移)                | `task.x_data`           | 通常为波数(cm⁻¹)           |
| `y` (强度)                  | `task.y_data`           | 光谱强度值                 |
| `explore_time`            | 积分时间                    | 对应原代码的积分概念            |
| `laser`                   | 激光功率                    | 对应原`power`            |
| `center_wave`             | 中心波长(nm)                | 对应原`wavenumber`（注意单位） |
