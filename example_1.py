import requests
import numpy as np
import time
import uuid
import threading
from typing import List, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
import logging
import socket
import traceback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """采集结果"""
    success: bool
    task_id: str
    x_data: Optional[np.ndarray] = None
    y_data: Optional[np.ndarray] = None
    error_msg: Optional[str] = None
    duration: float = 0.0


class CaptureTask:
    """任务对象"""
    
    def __init__(self, req_id: str, params: Dict, sequence: int = 0):
        self.req_id = req_id
        self.params = params  # {explore_time, integer, laser, center_wave}
        self.sequence = sequence  # 在队列中的序号
        self.status = "pending"  # pending/waiting/running/completed/failed
        self.result: Optional[CaptureResult] = None
        self.submit_time: Optional[float] = None
        self.complete_time: Optional[float] = None
        self.event = threading.Event()
        self.next_task: Optional['CaptureTask'] = None  # 链式队列指针
    
    def set_completed(self, x_data: np.ndarray, y_data: np.ndarray):
        self.status = "completed"
        self.complete_time = time.time()
        self.result = CaptureResult(
            success=True,
            task_id=self.req_id,
            x_data=x_data,
            y_data=y_data,
            duration=self.complete_time - self.submit_time if self.submit_time else 0
        )
        self.event.set()
        logger.info(f"✓ 任务 {self.sequence} [{self.req_id[:8]}] 完成")
    
    def set_failed(self, error_msg: str):
        self.status = "failed"
        self.complete_time = time.time()
        self.result = CaptureResult(
            success=False,
            task_id=self.req_id,
            error_msg=error_msg
        )
        logger.error(f"✗ 任务 {self.sequence} [{self.req_id[:8]}] 失败: {error_msg}")
        self.event.set()
    
    def wait(self, timeout: Optional[float] = None) -> CaptureResult:
        self.event.wait(timeout)
        return self.result


class CallbackHandler(BaseHTTPRequestHandler):
    """回调处理器 - 关键：触发队列下一个任务"""
    
    controller: Optional['RamanSpectrometerController'] = None  # 注入控制器引用
    
    def log_message(self, format, *args):
        logger.debug(f"[Callback] {args[0]}")
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            req_id = data.get("req_id")
            
            # 查找任务
            task = self.controller._find_task(req_id) if self.controller else None
            
            if not task:
                logger.warning(f"未知任务回调: {req_id}")
                self._respond(404, "Task not found")
                return
            
            # 检查仪器是否报告错误
            if data.get("code", 0) != 0:
                task.set_failed(data.get("msg", "Instrument error"))
                # 关键：即使失败也继续下一个，或根据策略停止
                self.controller._schedule_next(task)
                self._respond(0, "Error recorded")
                return
            
            # 解析数据
            payload = data.get("data", {})
            x_list = payload.get("x", [])
            y_list = payload.get("y", [])
            
            if not x_list or not y_list or len(x_list) != len(y_list):
                task.set_failed("Invalid data format")
                self.controller._schedule_next(task)
                self._respond(0, "Invalid data")
                return
            
            # 成功
            task.set_completed(np.array(x_list), np.array(y_list))
            
            # 关键：触发下一个任务
            self.controller._schedule_next(task)
            
            self._respond(0, "Success")
            
        except Exception as e:
            logger.error(f"回调处理异常: {traceback.format_exc()}")
            self._respond(500, f"Server error: {str(e)}")
    
    def _respond(self, code: int, msg: str):
        response_body = json.dumps({'code': code, 'msg': msg}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


class RamanSpectrometerController:
    
    def __init__(self, instrument_ip: str, instrument_port: int = 8088,
                 callback_port: int = 9000, callback_host: str = "0.0.0.0"):
        
        self.instrument_base = f"http://{instrument_ip}:{instrument_port}"
        self.capture_endpoint = f"{self.instrument_base}/raman/jy/capture"
        self.callback_url = f"http://{self._get_host_ip()}:{callback_port}/raman/jy/callback"
        
        self.callback_port = callback_port
        self.callback_host = callback_host
        
        # 任务管理
        self.task_queue: Queue[CaptureTask] = Queue()  # 待执行队列
        self.running_task: Optional[CaptureTask] = None  # 当前运行任务
        self.completed_tasks: List[CaptureTask] = []  # 已完成列表
        self.all_tasks: Dict[str, CaptureTask] = {}  # 全局索引
        
        self.lock = threading.Lock()
        self.queue_event = threading.Event()  # 队列调度信号
        
        # 启动回调服务器
        CallbackHandler.controller = self
        self._start_callback_server()
        
        # 启动队列调度线程
        self.scheduler_thread = threading.Thread(target=self._queue_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"控制器启动，回调地址: {self.callback_url}")
    
    def _get_host_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _start_callback_server(self):
        def run_server():
            server = HTTPServer((self.callback_host, self.callback_port), CallbackHandler)
            server.serve_forever()
        threading.Thread(target=run_server, daemon=True).start()
        time.sleep(0.5)
    
    def _find_task(self, req_id: str) -> Optional[CaptureTask]:
        """通过req_id查找任务"""
        with self.lock:
            return self.all_tasks.get(req_id)
    
    def _queue_scheduler(self):
        """后台调度线程：按顺序执行任务"""
        while True:
            # 等待队列信号
            self.queue_event.wait()
            self.queue_event.clear()
            
            with self.lock:
                # 检查当前是否有任务在运行
                if self.running_task is not None:
                    logger.debug(f"当前有任务运行中: {self.running_task.req_id[:8]}")
                    continue
                
                # 获取下一个任务
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    self.running_task = task
                else:
                    continue
            
            # 在锁外执行发送（避免阻塞回调）
            self._send_capture_command(task)
    
    def _send_capture_command(self, task: CaptureTask):
        """实际发送采集指令"""
        task.status = "running"
        task.submit_time = time.time()
        
        payload = {
            "req_id": task.req_id,
            "capture": {
                **task.params,
                "callback_url": self.callback_url
            }
        }
        
        try:
            response = requests.post(
                self.capture_endpoint,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            resp_data = response.json()
            
            if resp_data.get("code") != 0:
                # 仪器立即拒绝（如"当前有采集任务"）
                error_msg = resp_data.get("msg", "Unknown error")
                task.set_failed(f"仪器拒绝: {error_msg}")
                
                # 关键：如果是"忙"错误，重新排队稍后重试
                if "正在执行" in error_msg or "忙" in error_msg or "wait" in error_msg.lower():
                    logger.warning(f"任务 {task.sequence} 遇到仪器忙，重新排队...")
                    task.status = "pending"
                    task.event.clear()
                    with self.lock:
                        self.running_task = None
                        # 放回队列头部（优先重试）
                        # 使用临时列表实现插队
                        temp_list = list(self.task_queue.queue)
                        self.task_queue = Queue()
                        self.task_queue.put(task)
                        for item in temp_list:
                            self.task_queue.put(item)
                    # 延迟后触发重试
                    threading.Timer(1.0, lambda: self.queue_event.set()).start()
                    return
                
                # 其他错误：继续下一个
                self._schedule_next(task)
            else:
                logger.info(f"→ 任务 {task.sequence} [{task.req_id[:8]}] 已发送仪器")
                
        except Exception as e:
            task.set_failed(f"网络错误: {e}")
            self._schedule_next(task)
    
    def _schedule_next(self, completed_task: CaptureTask):
        """调度下一个任务（由回调触发）"""
        with self.lock:
            self.running_task = None
            self.completed_tasks.append(completed_task)
        
        logger.info(f"任务 {completed_task.sequence} 结束，触发下一个...")
        self.queue_event.set()  # 触发调度线程
    
    def add_task(self, explore_time: float, integer: int, 
                 laser: float, center_wave: float) -> CaptureTask:
        """
        添加任务到队列（不立即执行，等待调度）
        
        Returns:
            任务对象（可通过task.wait()等待完成）
        """
        req_id = str(uuid.uuid4())
        params = {
            "explore_time": explore_time,
            "integer": integer,
            "laser": laser,
            "center_wave": center_wave
        }
        
        with self.lock:
            sequence = len(self.all_tasks) + 1
            task = CaptureTask(req_id, params, sequence)
            self.all_tasks[req_id] = task
            self.task_queue.put(task)
        
        logger.info(f"+ 任务 {sequence} 加入队列: 积分{explore_time}s, 功率{laser}")
        
        # 触发调度（如果是第一个任务）
        if sequence == 1 or self.running_task is None:
            self.queue_event.set()
        
        return task
    
    def batch_capture(self, conditions: List[Tuple], 
                     wait_all: bool = True,
                     progress_callback: Optional[Callable[[int, int, str], None]] = None
                     ) -> List[CaptureResult]:
        """
        批量采集：构建队列并等待全部完成
        
        Args:
            conditions: [(explore_time, integer, laser, center_wave), ...]
            wait_all: 是否阻塞等待全部完成
        """
        total = len(conditions)
        tasks = []
        
        logger.info(f"构建采集队列: {total} 个任务")
        
        # 批量添加任务（此时都不执行，只入队）
        for et, it, ls, cw in conditions:
            task = self.add_task(et, it, ls, cw)
            tasks.append(task)
        
        if not wait_all:
            return tasks  # 返回任务列表，异步追踪
        
        # 阻塞等待全部完成
        results = []
        for i, task in enumerate(tasks):
            result = task.wait(timeout=60.0)  # 单个任务超时
            results.append(result)
            
            if progress_callback:
                progress_callback(i+1, total, 
                    f"完成 {task.params['laser']}/{task.params['center_wave']}")
        
        return results
    
    def get_queue_status(self) -> Dict:
        """获取当前队列状态"""
        with self.lock:
            return {
                "queued": self.task_queue.qsize(),
                "running": self.running_task.sequence if self.running_task else None,
                "completed": len(self.completed_tasks),
                "total": len(self.all_tasks)
            }
    
    def shutdown(self):
        """优雅关闭"""
        logger.info("关闭控制器...")
        # 清空队列，标记未完成
        with self.lock:
            while not self.task_queue.empty():
                task = self.task_queue.get()
                task.set_failed("Controller shutdown")
            if self.running_task and self.running_task.status == "running":
                self.running_task.set_failed("Controller shutdown")


# ==================== 与原代码兼容的包装器 ====================

class LegacyCompatibleController(RamanSpectrometerController):
    """兼容原代码风格：wavenumber_list/power_list遍历"""
    
    def __init__(self, instrument_ip: str, **kwargs):
        super().__init__(instrument_ip, **kwargs)
        self.default_explore_time = 1.0
        self.default_integer = 5
    
    def scan(self, wavenumber: float, power: float, 
             explore_time: Optional[float] = None,
             integer: Optional[int] = None,
             wait: bool = True) -> Union[np.ndarray, CaptureTask]:
        """
        兼容原scan函数
        如果wait=True返回numpy数组，否则返回task对象
        """
        et = explore_time or self.default_explore_time
        it = integer or self.default_integer
        
        task = self.add_task(
            explore_time=et,
            integer=it,
            laser=power,
            center_wave=wavenumber
        )
        
        if wait:
            result = task.wait(timeout=30.0)
            if result.success:
                return result.y_data
            else:
                raise RuntimeError(f"采集失败: {result.error_msg}")
        return task
    
    def batch_scan(self, wavenumber_list: List[float], 
                   power_list: List[float],
                   explore_time: Optional[float] = None,
                   integer: Optional[int] = None) -> List[np.ndarray]:
        """
        完全兼容原代码的双重循环
        自动队列化顺序执行，避免仪器忙冲突
        """
        et = explore_time or self.default_explore_time
        it = integer or self.default_integer
        
        # 构建所有条件组合
        conditions = []
        for wavenumber in wavenumber_list:
            for power in power_list:
                conditions.append((et, it, power, wavenumber))
        
        logger.info(f"批量扫描: {len(wavenumber_list)}×{len(power_list)}={len(conditions)} 个条件")
        
        # 使用队列化批量采集
        results = self.batch_capture(conditions, wait_all=True)
        
        # 提取y_data（与原代码兼容）
        spectra = []
        for r in results:
            if r.success:
                spectra.append(r.y_data)
            else:
                logger.warning(f"使用空数据填充失败项: {r.error_msg}")
                spectra.append(np.array([]))
        
        return spectra


# ==================== 使用示例 ====================

if __name__ == "__main__":
    
    instrument_ip = 
    controller = LegacyCompatibleController(
        instrument_ip=instrument_ip,
        callback_port=9000
    )
    
    try:
        print("\n" + "="*50)
        print("示例1: 自动队列化批量扫描")
        print("="*50)
        
        wavenumber_list = [800.0, 850.0, 900.0]
        power_list = [10.0, 50.0, 100.0]
        
        # 原代码写法，内部自动队列化顺序执行
        spectrum_list = controller.batch_scan(wavenumber_list, power_list)
        
        print(f"\n完成: {len(spectrum_list)} 个光谱")
        for i, sp in enumerate(spectrum_list):
            if len(sp) > 0:
                print(f"  [{i}] {len(sp)} 个点, 强度范围 [{sp.min():.1f}, {sp.max():.1f}]")
        
    finally:
        controller.shutdown()