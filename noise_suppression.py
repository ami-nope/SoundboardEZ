from __future__ import annotations

import ctypes
import ctypes.util
import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

RNNOISE_SAMPLE_RATE = 48000
RNNOISE_FRAME_SIZE = 480
RNNOISE_PCM_SCALE = 32768.0


@dataclass
class _Backend:
    name: str
    create_state: Callable[[], Any]
    process: Callable[[Any, np.ndarray], np.ndarray]
    destroy_state: Callable[[Any], None] | None = None


@dataclass
class _FallbackState:
    noise_floor: float = 0.008
    gain: float = 1.0


def _coerce_frame_candidate(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 1 and arr.size >= RNNOISE_FRAME_SIZE:
        return np.ascontiguousarray(arr[:RNNOISE_FRAME_SIZE], dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] >= RNNOISE_FRAME_SIZE and arr.shape[1] >= 1:
        return np.ascontiguousarray(arr[:RNNOISE_FRAME_SIZE, 0], dtype=np.float32)
    return None


def _extract_denoised_frame(result: Any, fallback_frame: np.ndarray) -> np.ndarray:
    direct = _coerce_frame_candidate(result)
    if direct is not None:
        return direct

    if isinstance(result, (tuple, list)):
        for item in result:
            candidate = _coerce_frame_candidate(item)
            if candidate is not None:
                return candidate

    return np.ascontiguousarray(fallback_frame[:RNNOISE_FRAME_SIZE], dtype=np.float32)


def _load_python_backend() -> tuple[_Backend | None, str | None]:
    candidates = (
        ("rnnoise", "RNNoise"),
        ("pyrnnoise", "RNNoise"),
    )

    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name} import failed: {exc}")
            continue

        state_cls = getattr(module, class_name, None)
        if state_cls is None:
            errors.append(f"{module_name} has no {class_name} class")
            continue

        method_name: str | None = None
        for candidate in ("process_frame", "process"):
            if callable(getattr(state_cls, candidate, None)):
                method_name = candidate
                break
        if method_name is None:
            errors.append(f"{module_name}.{class_name} has no process method")
            continue

        def create_state(cls=state_cls):
            return cls()

        def process(state: Any, frame: np.ndarray, proc_name: str = method_name) -> np.ndarray:
            frame_in = np.ascontiguousarray(frame, dtype=np.float32)
            frame_work = frame_in.copy()
            method = getattr(state, proc_name)
            result = method(frame_work)
            return _extract_denoised_frame(result, frame_work)

        def destroy_state(state: Any) -> None:
            for close_name in ("close", "destroy"):
                close_fn = getattr(state, close_name, None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                    break

        test_state = None
        try:
            test_state = create_state()
            test_out = process(test_state, np.zeros(RNNOISE_FRAME_SIZE, dtype=np.float32))
            if test_out.shape[0] != RNNOISE_FRAME_SIZE:
                raise RuntimeError("returned frame has wrong size")
        except Exception as exc:
            errors.append(f"{module_name} backend init failed: {exc}")
            continue
        finally:
            if test_state is not None:
                try:
                    destroy_state(test_state)
                except Exception:
                    pass

        return _Backend(
            name=f"{module_name}.{class_name}",
            create_state=create_state,
            process=process,
            destroy_state=destroy_state,
        ), None

    if not errors:
        return None, "No RNNoise Python binding found."
    return None, "; ".join(errors)


def _load_ctypes_backend() -> tuple[_Backend | None, str | None]:
    candidates: list[str] = []
    discovered = ctypes.util.find_library("rnnoise")
    if discovered:
        candidates.append(discovered)
    candidates.extend(
        [
            "rnnoise",
            "rnnoise.dll",
            "librnnoise.so",
            "librnnoise.dylib",
        ]
    )

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique_candidates.append(item)

    errors: list[str] = []
    for lib_name in unique_candidates:
        try:
            lib = ctypes.CDLL(lib_name)
        except OSError as exc:
            errors.append(f"{lib_name}: {exc}")
            continue

        try:
            create_fn = lib.rnnoise_create
            create_fn.restype = ctypes.c_void_p
            create_fn.argtypes = [ctypes.c_void_p]

            process_fn = lib.rnnoise_process_frame
            process_fn.restype = ctypes.c_float
            process_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]

            destroy_fn = lib.rnnoise_destroy
            destroy_fn.restype = None
            destroy_fn.argtypes = [ctypes.c_void_p]
        except Exception as exc:
            errors.append(f"{lib_name}: invalid RNNoise symbols ({exc})")
            continue

        def create_state(cf=create_fn):
            state = cf(None)
            if not state:
                raise RuntimeError("rnnoise_create returned null state")
            return state

        def process(state: Any, frame: np.ndarray, pf=process_fn) -> np.ndarray:
            frame_in = np.ascontiguousarray(frame, dtype=np.float32)
            out_frame = np.empty(RNNOISE_FRAME_SIZE, dtype=np.float32)
            # RNNoise C API expects 16-bit PCM scale in float buffers.
            # Internal app audio is normalized [-1.0, 1.0], so scale in/out.
            scaled_in = np.ascontiguousarray(frame_in * RNNOISE_PCM_SCALE, dtype=np.float32)
            out_ptr = out_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            in_ptr = scaled_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            pf(state, out_ptr, in_ptr)
            out_norm = out_frame / RNNOISE_PCM_SCALE
            return np.clip(out_norm, -1.0, 1.0).astype(np.float32, copy=False)

        def destroy_state(state: Any, df=destroy_fn) -> None:
            if state:
                df(state)

        test_state = None
        try:
            test_state = create_state()
            test_out = process(test_state, np.zeros(RNNOISE_FRAME_SIZE, dtype=np.float32))
            if test_out.shape[0] != RNNOISE_FRAME_SIZE:
                raise RuntimeError("returned frame has wrong size")
        except Exception as exc:
            errors.append(f"{lib_name}: backend init failed ({exc})")
            continue
        finally:
            if test_state is not None:
                try:
                    destroy_state(test_state)
                except Exception:
                    pass

        return _Backend(
            name=f"ctypes:{lib_name}",
            create_state=create_state,
            process=process,
            destroy_state=destroy_state,
        ), None

    if not errors:
        return None, "RNNoise native library not found."
    return None, "; ".join(errors)


def _load_backend() -> tuple[_Backend | None, str | None]:
    py_backend, py_error = _load_python_backend()
    if py_backend is not None:
        return py_backend, None

    c_backend, c_error = _load_ctypes_backend()
    if c_backend is not None:
        return c_backend, None

    details = " | ".join(err for err in (py_error, c_error) if err)
    if not details:
        details = "RNNoise backend unavailable."
    return _load_builtin_backend(), details


def _load_builtin_backend() -> _Backend:
    def create_state() -> _FallbackState:
        return _FallbackState()

    def process(state: _FallbackState, frame: np.ndarray) -> np.ndarray:
        frame_in = np.ascontiguousarray(frame, dtype=np.float32)
        if frame_in.size == 0:
            return frame_in

        rms = float(np.sqrt(np.mean(frame_in * frame_in)) + 1e-8)
        floor_target = min(rms, state.noise_floor * 4.0 + 0.02)
        state.noise_floor = max(0.002, min(0.08, state.noise_floor * 0.98 + floor_target * 0.02))
        threshold = max(0.006, state.noise_floor * 2.4)

        if rms <= threshold:
            target_gain = max(0.12, min(1.0, rms / (threshold + 1e-8)))
        else:
            target_gain = 1.0

        # Faster attenuation than recovery keeps hiss/noise floor lower.
        if target_gain < state.gain:
            state.gain = state.gain * 0.55 + target_gain * 0.45
        else:
            state.gain = state.gain * 0.92 + target_gain * 0.08

        out = frame_in * np.float32(state.gain)
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    return _Backend(
        name="builtin.noise_gate",
        create_state=create_state,
        process=process,
    )


class MicNoiseSuppressor:
    def __init__(self, sample_rate: int = RNNOISE_SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_size = RNNOISE_FRAME_SIZE
        self.available = False
        self.backend_name = ""
        self.error_message: str | None = None
        self.debug_error: str | None = None
        self._backend: _Backend | None = None
        self._states: list[Any] = []
        self._tails: list[np.ndarray] = []

        if self.sample_rate != RNNOISE_SAMPLE_RATE:
            self.error_message = (
                f"RNNoise requires {RNNOISE_SAMPLE_RATE} Hz input, got {self.sample_rate} Hz."
            )
            return

        backend, error = _load_backend()
        if backend is None:
            self.error_message = (
                "RNNoise backend unavailable. Install rnnoise/pyrnnoise, or install a system librnnoise."
            )
            self.debug_error = error
            return

        self._backend = backend
        self.backend_name = backend.name
        self.available = True
        if error:
            self.debug_error = error

    def _clear_states(self) -> None:
        if self._backend is None or self._backend.destroy_state is None:
            self._states.clear()
            self._tails.clear()
            return
        for state in self._states:
            try:
                self._backend.destroy_state(state)
            except Exception:
                pass
        self._states.clear()
        self._tails.clear()

    def reset(self) -> None:
        self._clear_states()

    def close(self) -> None:
        self._clear_states()

    def _ensure_states(self, channels: int) -> None:
        if self._backend is None:
            return
        target = max(1, int(channels))
        if len(self._states) == target and len(self._tails) == target:
            return
        self._clear_states()
        self._states = [self._backend.create_state() for _ in range(target)]
        self._tails = [np.zeros(0, dtype=np.float32) for _ in range(target)]

    def _process_channel(self, samples: np.ndarray, state: Any, channel_idx: int) -> np.ndarray:
        if self._backend is None:
            return np.ascontiguousarray(samples, dtype=np.float32)
        samples_in = np.ascontiguousarray(samples, dtype=np.float32)
        total = int(samples_in.shape[0])
        if total == 0:
            return np.ascontiguousarray(samples_in, dtype=np.float32)

        if channel_idx < len(self._tails):
            tail = self._tails[channel_idx]
        else:
            tail = np.zeros(0, dtype=np.float32)
        carry_samples = int(tail.shape[0])

        if carry_samples > 0:
            merged = np.concatenate((tail, samples_in))
        else:
            merged = samples_in
        merged = np.ascontiguousarray(merged, dtype=np.float32)
        merged_total = int(merged.shape[0])
        full_len = (merged_total // self.frame_size) * self.frame_size

        if channel_idx < len(self._tails):
            self._tails[channel_idx] = np.ascontiguousarray(merged[full_len:], dtype=np.float32)

        if full_len <= 0:
            return np.ascontiguousarray(samples_in, dtype=np.float32)

        processed_full = np.empty(full_len, dtype=np.float32)
        offset = 0
        while offset < full_len:
            end = offset + self.frame_size
            frame = merged[offset:end]
            processed = self._backend.process(state, frame)
            processed = np.ascontiguousarray(processed, dtype=np.float32)
            if processed.shape[0] < self.frame_size:
                repaired = np.zeros(self.frame_size, dtype=np.float32)
                repaired[: processed.shape[0]] = processed
                processed = repaired
            processed_full[offset:end] = processed[: self.frame_size]
            offset = end

        out_start = carry_samples
        out_end = out_start + total
        if out_end <= full_len:
            return np.ascontiguousarray(processed_full[out_start:out_end], dtype=np.float32)

        out = np.empty(total, dtype=np.float32)
        processed_count = max(0, full_len - out_start)
        if processed_count > 0:
            out[:processed_count] = processed_full[out_start:full_len]
        if processed_count < total:
            raw_needed = total - processed_count
            raw_start = full_len
            raw_end = raw_start + raw_needed
            out[processed_count:] = merged[raw_start:raw_end]
        return out

    def process_mic_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(audio_frame, dtype=np.float32)
        if not self.available or self._backend is None:
            return np.ascontiguousarray(frame, dtype=np.float32)

        if frame.ndim == 1:
            self._ensure_states(1)
            return self._process_channel(frame, self._states[0], 0)

        if frame.ndim == 2:
            channels = int(frame.shape[1])
            self._ensure_states(channels)
            out = np.empty_like(frame, dtype=np.float32)
            for ch in range(channels):
                out[:, ch] = self._process_channel(frame[:, ch], self._states[ch], ch)
            return out

        return np.ascontiguousarray(frame, dtype=np.float32)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
