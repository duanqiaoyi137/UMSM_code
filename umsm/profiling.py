"""Runtime and peak-memory logging per pipeline stage.

Peak memory is the peak resident set size (RSS) of the process sampled by a background thread
every `interval_s` seconds; stages are measured as windows on that signal, so nested stages work.
Optionally tracemalloc reports the peak of Python/NumPy allocations of top-level stages
(slower, off by default). On Windows the OS-tracked peak working set is reported as well.
"""
from __future__ import annotations

import threading
import time
import tracemalloc
from contextlib import contextmanager

import psutil

_MB = 1024 * 1024


class _RssSampler(threading.Thread):
    def __init__(self, interval_s: float):
        # 输入：interval_s；输出：None；初始化实例及校验参数；继承：_RssSampler(threading.Thread)。
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.proc = psutil.Process()
        self.windows: list[dict] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.global_peak = self.rss()

    def rss(self) -> int:
        # 输入：实例状态；输出：当前内存字节数；继承：_RssSampler(threading.Thread)。
        return self.proc.memory_info().rss

    def sample(self) -> int:
        # 输入：实例状态；输出：None；采集内存峰值；继承：_RssSampler(threading.Thread)。
        cur = self.rss()
        with self.lock:
            self.global_peak = max(self.global_peak, cur)
            for w in self.windows:
                if cur > w["peak"]:
                    w["peak"] = cur
        return cur

    def run(self):
        # 输入：实例状态；输出：None；后台持续采样；继承：_RssSampler(threading.Thread)。
        while not self._stop_event.wait(self.interval_s):
            self.sample()

    def open_window(self) -> dict:
        # 输入：实例状态；输出：内存统计窗口；继承：_RssSampler(threading.Thread)。
        cur = self.sample()
        w = {"start": cur, "peak": cur}
        with self.lock:
            self.windows.append(w)
        return w

    def close_window(self, w: dict) -> dict:
        # 输入：w；输出：关闭后的内存窗口统计字典；继承：_RssSampler(threading.Thread)。
        self.sample()
        with self.lock:
            self.windows = [x for x in self.windows if x is not w]  # nested windows may hold equal values
        return w

    def stop(self):
        # 输入：实例状态；输出：None；停止资源采样；继承：_RssSampler(threading.Thread)。
        self._stop_event.set()


class Profiler:
    def __init__(self, interval_s: float = 0.02, use_tracemalloc: bool = False):
        # 输入：interval_s、use_tracemalloc；输出：None；初始化实例及校验参数；继承：Profiler(object)。
        self.sampler = _RssSampler(interval_s)
        self.use_tracemalloc = use_tracemalloc
        self.records: list[dict] = []
        self._depth = 0
        self._t0 = None
        self._compact = None

    def start(self):
        # 输入：实例状态；输出：启动后的 profiler 自身；继承：Profiler(object)。
        self.sampler.start()
        if self.use_tracemalloc:
            tracemalloc.start()
        self._t0 = time.perf_counter()
        self.baseline_rss_mb = self.sampler.rss() / _MB
        return self

    def stop(self) -> dict:
        # 输入：实例状态；输出：整轮资源统计字典；继承：Profiler(object)。
        total = time.perf_counter() - self._t0
        self.sampler.sample()
        self.sampler.stop()
        self.sampler.join()
        if self.use_tracemalloc:
            tracemalloc.stop()
        mem = self.sampler.proc.memory_info()
        summary = {
            "total_time_s": total,
            "baseline_rss_mb": self.baseline_rss_mb,
            "peak_rss_mb": self.sampler.global_peak / _MB,
            "peak_rss_minus_baseline_mb": self.sampler.global_peak / _MB - self.baseline_rss_mb,
        }
        if hasattr(mem, "peak_wset"):  # Windows
            summary["os_peak_working_set_mb"] = mem.peak_wset / _MB
        return summary

    @contextmanager
    def stage(self, name: str, **tags):
        # 输入：name、**tags；输出：记录阶段耗时与内存的上下文管理器；继承：Profiler(object)。
        top_level = self._depth == 0
        self._depth += 1
        w = self.sampler.open_window()
        if self.use_tracemalloc and top_level:
            tracemalloc.reset_peak()
        t = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t
            self.sampler.close_window(w)
            self._depth -= 1
            rec = {"stage": name, "time_s": elapsed, "rss_start_mb": w["start"] / _MB,
                   "rss_peak_mb": w["peak"] / _MB, "rss_peak_increase_mb": (w["peak"] - w["start"]) / _MB,
                   "depth": self._depth, **tags}
            if self.use_tracemalloc and top_level:
                rec["traced_peak_mb"] = tracemalloc.get_traced_memory()[1] / _MB
            if self._compact is None:
                self.records.append(rec)
            else:
                key=(name,tuple(sorted((k,str(v)) for k,v in tags.items())),self._depth)
                if key not in self._compact:
                    self._compact[key]={**rec,'calls':1}
                else:
                    total=self._compact[key]
                    total['time_s']+=rec['time_s']; total['calls']+=1
                    for field in ('rss_peak_mb','rss_peak_increase_mb','traced_peak_mb'):
                        if field in rec: total[field]=max(total.get(field,0.),rec[field])

    @contextmanager
    def compact(self):
        # 输入：在线阶段；输出：上下文，按阶段合并轮次记录，避免日志空间随 R 增长；继承：Profiler(object)。
        previous=self._compact
        if previous is not None:
            yield
            return
        self._compact={}
        try:
            yield
        finally:
            self.records.extend(self._compact.values())
            self._compact=previous

    def stage_totals(self) -> dict[str, dict]:
        # 输入：实例状态；输出：按阶段累计耗时；继承：Profiler(object)。
        """Aggregate records with the same stage name (e.g. over the sets of Real/Align)."""
        out: dict[str, dict] = {}
        for r in self.records:
            agg = out.setdefault(r["stage"], {"time_s": 0.0, "rss_peak_mb": 0.0, "rss_peak_increase_mb": 0.0, "calls": 0})
            agg["time_s"] += r["time_s"]
            agg["rss_peak_mb"] = max(agg["rss_peak_mb"], r["rss_peak_mb"])
            agg["rss_peak_increase_mb"] = max(agg["rss_peak_increase_mb"], r["rss_peak_increase_mb"])
            agg["calls"] += r.get('calls',1)
        return out


class _NullProfiler:
    @contextmanager
    def compact(self):
        # 输入：在线阶段；输出：不记录数据的上下文；继承：_NullProfiler(object)。
        yield

    @contextmanager
    def stage(self, name: str, **tags):
        # 输入：name、**tags；输出：记录阶段耗时与内存的上下文管理器；继承：_NullProfiler(object)。
        yield


NULL_PROFILER = _NullProfiler()
