"""
Hot-swap model registry.

A "promotion" writes the new model's path into an ACTIVE_VERSION pointer
file. A background watcher thread (using `watchdog`) polls this file for
changes and, when it changes, calls `classifier.reload(new_path)` — this
swaps the model in place inside the already-running consumer process, with
no restart and no interruption to in-flight message processing.
"""
import time
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

DEFAULT_REGISTRY_DIR = Path("models")
DEFAULT_POINTER_FILE = DEFAULT_REGISTRY_DIR / "ACTIVE_VERSION"


def promote(model_dir: str, pointer_file: Path = DEFAULT_POINTER_FILE) -> None:
    """
    Marks `model_dir` as the currently active model version. This is the
    only step a deployment pipeline needs to call after training/validating
    a new model — every running consumer picks it up automatically.
    """
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text(str(Path(model_dir).resolve()))
    print(f"[registry] Promoted '{model_dir}' to active. Pointer -> {pointer_file}")


def get_active_model_dir(pointer_file: Path = DEFAULT_POINTER_FILE, fallback: str = None) -> str:
    if pointer_file.exists():
        return pointer_file.read_text().strip()
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"No active model pointer at {pointer_file} and no fallback provided. "
        f"Call promote(<model_dir>) first."
    )


class _PointerFileHandler(FileSystemEventHandler):
    def __init__(self, pointer_file: Path, on_change_callback):
        self.pointer_file = str(pointer_file.resolve())
        self.on_change_callback = on_change_callback
        self._last_value = None

    def _maybe_fire(self):
        try:
            value = Path(self.pointer_file).read_text().strip()
        except FileNotFoundError:
            return
        if value != self._last_value:
            self._last_value = value
            self.on_change_callback(value)

    def on_modified(self, event):
        if Path(event.src_path).resolve() == Path(self.pointer_file).resolve():
            self._maybe_fire()

    def on_created(self, event):
        self.on_modified(event)


class HotSwapWatcher:
    """
    Watches the ACTIVE_VERSION pointer file and calls `classifier.reload(...)`
    whenever it changes, in a background thread. Use as a context manager
    around the consumer's main loop.
    """

    def __init__(self, classifier, pointer_file: Path = DEFAULT_POINTER_FILE,
                 poll_interval_s: float = 1.0):
        self.classifier = classifier
        self.pointer_file = pointer_file
        self.poll_interval_s = poll_interval_s
        self._observer = None
        self._stop_flag = threading.Event()
        self._poll_thread = None

    def _handle_change(self, new_model_dir: str):
        print(f"[hot-swap] Detected new active model: {new_model_dir}. Reloading...")
        t0 = time.perf_counter()
        self.classifier.reload(new_model_dir)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[hot-swap] Reload complete in {dt:.1f}ms. "
              f"Now serving model_version={self.classifier.model_version}")

    def start(self):
        # Simple, dependency-light polling loop — robust across filesystems
        # (works with Docker bind mounts / NFS where inotify events can be
        # unreliable). Swap for the watchdog Observer if you prefer
        # event-driven notification on a local filesystem.
        def _poll_loop():
            last_value = None
            while not self._stop_flag.is_set():
                if self.pointer_file.exists():
                    value = self.pointer_file.read_text().strip()
                    if value != last_value:
                        if last_value is not None:  # skip the very first read
                            self._handle_change(value)
                        last_value = value
                time.sleep(self.poll_interval_s)

        self._poll_thread = threading.Thread(target=_poll_loop, daemon=True)
        self._poll_thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
