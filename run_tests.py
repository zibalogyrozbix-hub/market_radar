"""Dependency-free test runner (use `pytest` when available)."""
import importlib, inspect, pkgutil, sys, time, traceback
sys.path.insert(0, ".")
import tests
failed = passed = skipped = 0
for m in pkgutil.iter_modules(tests.__path__):
    if not m.name.startswith("test_"):
        continue
    mod = importlib.import_module(f"tests.{m.name}")
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        t = time.time()
        try:
            r = fn()
            if r is True:
                skipped += 1; print(f"SKIP {m.name}::{name}")
            else:
                passed += 1; print(f"PASS {m.name}::{name} ({time.time()-t:.1f}s)")
        except Exception:
            failed += 1; print(f"FAIL {m.name}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
