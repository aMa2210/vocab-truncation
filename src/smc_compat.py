"""Runtime fix for a bug in the released genlm-backend 0.3.0 (vLLM backend).

`genlm.backend.llm.vllm.AsyncVirtualLM` defines `_resolve` twice: an instance method
`_resolve(self, rows)` (the capture callback, which hands each forward's log-probability
rows to the waiting futures) and, later in the class body, a
`@staticmethod _resolve(loop, future, result=None, exc=None)` (the future setter). The
second definition replaces the first, so the one caller of the capture path -- the
module-level `_weak_resolve`, which calls `target._resolve(rows)` -- crashes with
"missing 1 required positional argument: 'future'" on the very first forward pass.

The fix restores the lost method under its own name, `_resolve_rows`, and points
`_weak_resolve` at it. The staticmethod and its call sites are left untouched.
`from_name` looks `_weak_resolve` up as a module global when it builds each model, so
patching the module attribute is enough. The patch applies only while `_resolve` is still
a staticmethod, so it becomes a no-op once upstream fixes the bug.

Call `apply_v2()` before the first model is built (src/run_smc.py does this at import).
"""


def apply_v2():
    import inspect, weakref
    import genlm.backend.llm.vllm as vm
    cls = vm.AsyncVirtualLM
    if not isinstance(inspect.getattr_static(cls, "_resolve"), staticmethod):
        return

    def _resolve_rows(self, rows):
        """(engine thread) Capture callback: resolve the forward's pending futures,
        one callback per event loop. Body of upstream's shadowed `_resolve(self, rows)`."""
        by_loop = {}
        for req_id, row in rows.items():
            waiters = self._pending.pop(req_id, None)
            if not waiters:
                continue
            self._served += 1
            for i, (future, loop) in enumerate(waiters):
                by_loop.setdefault(loop, []).append((future, row if i == 0 else row.clone()))
        for loop, items in by_loop.items():
            def _set(items=items):
                for future, result in items:
                    if not future.done():
                        future.set_result(result)
            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:
                pass                                   # the loop closed; no reader

    def _weak_resolve(inst):
        """Route captured rows to `inst` without keeping it alive (as upstream does)."""
        ref = weakref.ref(inst)

        def resolve(rows):
            target = ref()
            if target is not None:
                target._resolve_rows(rows)
        return resolve

    cls._resolve_rows = _resolve_rows
    vm._weak_resolve = _weak_resolve
