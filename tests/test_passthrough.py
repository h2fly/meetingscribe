"""Tests for the bypass watchdog: apps that play straight into BlackHole
(skipping the Multi-Output) get BlackHole forwarded to the physical output.

CoreAudio and PortAudio are faked; the real-device behaviour was verified
by hand (see CLAUDE.md, _BypassMonitor)."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import meetingscribe as ms

TRANSPORTS = {
    "BlackHole 2ch": "virt",
    "多输出设备": "grup",
    "ZoomAudioDevice": "virt",
    "External Headphones": "bltn",
    "MacBook Air Speakers": "bltn",
}
BYPASSER = {"pid": 101, "bundle": "cn.com.langeasy.ZdcQlisten",
            "devices": ["BlackHole 2ch"]}
NORMAL = {"pid": 102, "bundle": "com.apple.Music",
          "devices": ["BlackHole 2ch", "External Headphones"]}


class TestBypassingProcesses:
    def test_blackhole_only_is_bypassing(self):
        assert ms._bypassing_processes([BYPASSER, NORMAL], "BlackHole 2ch",
                                       TRANSPORTS, own_pid=1) == [BYPASSER]

    def test_through_multi_output_is_not(self):
        assert ms._bypassing_processes([NORMAL], "BlackHole 2ch",
                                       TRANSPORTS, own_pid=1) == []

    def test_own_process_excluded(self):
        assert ms._bypassing_processes([BYPASSER], "BlackHole 2ch",
                                       TRANSPORTS, own_pid=101) == []

    def test_other_virtual_device_only_is_not_bypassing_capture(self):
        zoom = {"pid": 103, "bundle": "us.zoom.xos", "devices": ["ZoomAudioDevice"]}
        assert ms._bypassing_processes([zoom], "BlackHole 2ch",
                                       TRANSPORTS, own_pid=1) == []


class _FakeProc:
    def __init__(self, first_line="ready\n", final="0\n", pid=4242):
        import io
        self.stdout = io.StringIO(first_line)
        self.pid = pid
        self.final = final
        self.killed = False
        self.sent = None

    def communicate(self, input=None, timeout=None):
        self.sent = input
        return self.final, "boom: no such device\n"

    def kill(self):
        self.killed = True


class TestPassthroughChild:
    """The stream runs in a child process (GIL isolation); spawn is faked."""

    def _patch(self, monkeypatch, proc):
        calls = {}

        def popen(argv, **kw):
            calls["argv"] = argv
            return proc

        monkeypatch.setattr(ms.subprocess, "Popen", popen)
        import select
        monkeypatch.setattr(select, "select", lambda r, w, x, t: (r, [], []))
        return calls

    def test_spawns_child_with_devices_and_waits_for_ready(self, monkeypatch):
        proc = _FakeProc()
        calls = self._patch(monkeypatch, proc)
        pt = ms._BlackHolePassthrough("BlackHole 2ch", "External Headphones")
        pt.start()
        assert calls["argv"][1:2] == ["-c"]
        assert calls["argv"][3:] == ["BlackHole 2ch", "External Headphones", "48000"]
        assert pt.pid == 4242

    def test_child_that_never_gets_ready_raises_with_its_error(self, monkeypatch):
        proc = _FakeProc(first_line="")
        self._patch(monkeypatch, proc)
        pt = ms._BlackHolePassthrough("BlackHole 2ch", "External Headphones")
        with pytest.raises(RuntimeError, match="no such device"):
            pt.start()
        assert proc.killed and pt.pid is None

    def test_stop_closes_stdin_and_reads_glitches(self, monkeypatch):
        proc = _FakeProc(final="3\n")
        self._patch(monkeypatch, proc)
        pt = ms._BlackHolePassthrough("BlackHole 2ch", "External Headphones")
        pt.start()
        pt.stop()
        assert proc.sent == ""           # EOF on stdin is the child's stop signal
        assert pt.glitches == 3 and pt.pid is None
        pt.stop()                        # idempotent

    def test_child_script_copies_input_to_output(self):
        # The callback is defined inside the child source; run it here.
        import types as _t
        fake_sd = _t.ModuleType("sounddevice")

        class _Stream:
            def __init__(self, **kw):
                _Stream.cb = kw["callback"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_sd.Stream = _Stream
        import io
        old = (sys.modules.get("sounddevice"), sys.argv, sys.stdin, sys.stdout)
        sys.modules["sounddevice"] = fake_sd
        sys.argv = ["x", "BlackHole 2ch", "External Headphones", "48000"]
        sys.stdin, sys.stdout = io.StringIO(""), io.StringIO()
        try:
            exec(ms._PASSTHRU_CHILD, {})
            indata = np.random.rand(16, 2).astype(np.float32)
            out = np.zeros_like(indata)
            _Stream.cb(indata, out, 16, None, None)
            assert np.array_equal(out, indata)
        finally:
            sys.modules["sounddevice"], sys.argv, sys.stdin, sys.stdout = old


class _FakePT:
    started: list = []

    def __init__(self, src, dst, samplerate=48000):
        self.src, self.dst = src, dst
        self.stopped = False
        self.pid = 9999

    def start(self):
        _FakePT.started.append(self.dst)

    def stop(self):
        self.stopped = True


class _FakeListener:
    instances: list = []

    def __init__(self, wake):
        self.wake = wake
        self.installed = self.removed = False
        self.syncs = 0
        _FakeListener.instances.append(self)

    def install(self):
        self.installed = True
        return True

    def sync(self):
        self.syncs += 1

    def remove(self):
        self.removed = True


@pytest.fixture
def world(monkeypatch):
    state = types.SimpleNamespace(procs=[], output="External Headphones",
                                  now=1000.0)
    monkeypatch.setattr(ms, "_ProcessAudioListener", _FakeListener)
    _FakeListener.instances = []
    monkeypatch.setattr(ms, "_audio_output_processes", lambda: list(state.procs))
    monkeypatch.setattr(ms, "_coreaudio_device_info", lambda: dict(TRANSPORTS))
    monkeypatch.setattr(ms, "_get_system_output_device", lambda: state.output)
    monkeypatch.setattr(ms, "_BlackHolePassthrough", _FakePT)
    monkeypatch.setattr(ms, "_log", lambda cat, msg: None)
    _FakePT.started = []
    return state


def _monitor():
    return ms._BypassMonitor("BlackHole 2ch", "MacBook Air Speakers")


class TestBypassMonitor:
    def test_forwarding_starts_on_first_check(self, world):
        world.procs = [BYPASSER]
        mon = _monitor()
        assert mon._tick()
        assert mon._pt.dst == "External Headphones"

    def test_no_forwarding_for_normal_players(self, world):
        world.procs = [NORMAL]
        mon = _monitor()
        mon._tick()
        assert mon._pt is None

    def test_follows_output_change(self, world):
        world.procs = [BYPASSER]
        mon = _monitor()
        mon._tick()
        first = mon._pt
        world.output = "MacBook Air Speakers"     # headphones unplugged
        mon._tick()
        assert first.stopped and mon._pt.dst == "MacBook Air Speakers"

    def test_never_targets_a_virtual_device(self, world):
        world.procs = [BYPASSER]
        world.output = "多输出设备"               # dOut is the aggregate
        mon = _monitor()
        mon._tick()
        assert mon._pt.dst == "MacBook Air Speakers"   # falls back to physical

    def test_stops_after_idle_grace_not_before(self, world, monkeypatch):
        monkeypatch.setattr(ms.time, "time", lambda: world.now)
        world.procs = [BYPASSER]
        mon = _monitor()
        mon._tick()
        world.procs = []                          # paused
        mon._tick()
        world.now += ms._PASSTHRU_IDLE_GRACE_SEC - 0.5
        mon._tick()
        assert mon._pt is not None                # a pause is ridden out
        world.now += 1.0
        mon._tick()
        assert mon._pt is None

    def test_idle_shortens_the_wait_so_the_grace_is_honoured(self, world, monkeypatch):
        monkeypatch.setattr(ms.time, "time", lambda: world.now)
        mon = _monitor()
        assert mon._next_timeout() == ms._PASSTHRU_SAFETY_TIMEOUT_SEC
        mon._idle_since = world.now - 1.0
        assert mon._next_timeout() == pytest.approx(ms._PASSTHRU_IDLE_GRACE_SEC - 1.0)

    def test_resumes_after_pause_without_restart(self, world):
        world.procs = [BYPASSER]
        mon = _monitor()
        mon._tick()
        world.procs = []
        mon._tick()
        world.procs = [BYPASSER]
        mon._tick()
        assert _FakePT.started == ["External Headphones"]   # same stream kept

    def test_api_unavailable_disables_check(self, world, monkeypatch):
        monkeypatch.setattr(ms, "_audio_output_processes", lambda: None)
        assert _monitor()._tick() is False

    def test_stop_closes_forwarding(self, world):
        world.procs = [BYPASSER]
        mon = _monitor()
        mon._tick()
        pt = mon._pt
        mon.stop()
        assert pt.stopped and mon._pt is None


class TestStartBypassMonitor:
    def _plan(self, sys_source="BlackHole 2ch"):
        return ms.AudioPlan(sys_source_name=sys_source,
                            restore_output_name="External Headphones")

    def test_disabled_by_config(self, world):
        cfg = {"audio": {"passthrough_bypass": False}}
        assert ms._start_bypass_monitor(cfg, self._plan()) is None

    def test_needs_a_system_audio_source(self, world):
        assert ms._start_bypass_monitor({}, self._plan(sys_source=None)) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_start_runs_the_first_check_synchronously(self, world):
        world.procs = [BYPASSER]
        mon = ms._start_bypass_monitor({}, self._plan())
        try:
            assert mon is not None and mon._pt is not None   # before any poll
        finally:
            ms._stop_bypass_monitor(mon)


class TestEventDriven:
    def test_routing_event_triggers_a_check(self, world):
        import time as _time
        mon = _monitor()
        mon.start()                               # nothing bypassing yet
        try:
            assert mon._pt is None
            world.procs = [BYPASSER]              # the app re-routes...
            _FakeListener.instances[-1].wake.set()   # ...CoreAudio notifies
            deadline = _time.time() + 2
            while mon._pt is None and _time.time() < deadline:
                _time.sleep(0.02)
            assert mon._pt is not None and mon._pt.dst == "External Headphones"
            assert _FakeListener.instances[-1].syncs >= 1   # new processes watched
        finally:
            mon.stop()

    def test_no_event_no_check(self, world, monkeypatch):
        import time as _time
        calls = []
        monkeypatch.setattr(ms, "_audio_output_processes",
                            lambda: calls.append(1) or [])
        mon = _monitor()
        mon.start()
        try:
            _time.sleep(1.5)                      # the old design polled here
            assert len(calls) == 1                # only the start-time check
        finally:
            mon.stop()

    def test_stop_removes_listeners(self, world):
        mon = _monitor()
        mon.start()
        mon.stop()
        assert _FakeListener.instances[-1].removed
