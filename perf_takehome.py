"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(
        self,
        *,
        alu: list[tuple] | None = None,
        valu: list[tuple] | None = None,
        load: list[tuple] | None = None,
        store: list[tuple] | None = None,
        flow: list[tuple] | None = None,
        debug: list[tuple] | None = None,
    ):
        bundle = {}
        if alu:
            bundle["alu"] = alu
        if valu:
            bundle["valu"] = valu
        if load:
            bundle["load"] = load
        if store:
            bundle["store"] = store
        if flow:
            bundle["flow"] = flow
        if debug:
            bundle["debug"] = debug
        if bundle:
            self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    # --------------------
    # Optimized kernel below
    # --------------------

    @dataclass
    class _Task:
        engine: str
        slot: tuple
        reads: tuple[int, ...] = ()
        writes: tuple[int, ...] = ()
        preds: list[int] = field(default_factory=list)
        succs: list[int] = field(default_factory=list)
        unsatisfied: int = 0
        earliest: int = 0
        scheduled: bool = False
        cycle: int | None = None
        cp: int = 1  # critical-path estimate

    def _vec_addrs(self, base: int) -> tuple[int, ...]:
        return tuple(base + i for i in range(VLEN))

    def _mk_task(
        self,
        tasks: list[_Task],
        last_writer: dict[int, int],
        last_reader: dict[int, int],
        *,
        engine: str,
        slot: tuple,
        reads: tuple[int, ...] = (),
        writes: tuple[int, ...] = (),
    ) -> int:
        preds_set: set[int] = set()
        for addr in reads:
            if addr in last_writer:
                preds_set.add(last_writer[addr])
        for addr in writes:
            if addr in last_writer:
                preds_set.add(last_writer[addr])
        for addr in writes:
            if addr in last_reader:
                preds_set.add(last_reader[addr])
        tid = len(tasks)
        task = KernelBuilder._Task(
            engine=engine, slot=slot, reads=reads, writes=writes, preds=sorted(preds_set)
        )
        tasks.append(task)
        for p in task.preds:
            tasks[p].succs.append(tid)
        for addr in writes:
            last_writer[addr] = tid
        for addr in reads:
            last_reader[addr] = tid
        for addr in writes:
            last_reader.pop(addr, None)
        return tid

    def _compute_cp(self, tasks: list[_Task]) -> None:
        # Reverse-topo-ish: tasks are appended in dependency-respecting order, so reverse is fine.
        for tid in range(len(tasks) - 1, -1, -1):
            task = tasks[tid]
            if not task.succs:
                task.cp = 1
            else:
                task.cp = 1 + max(tasks[s].cp for s in task.succs)

    def _schedule(
        self, tasks: list[_Task], *, dummy_dest: int | None = None
    ) -> list[dict[str, list[tuple]]]:
        if dummy_dest is None:
            raise RuntimeError("dummy_dest is required")

        engine_order = ("load", "store", "flow", "valu", "alu")

        n = len(tasks)
        unsatisfied = [len(t.preds) for t in tasks]
        earliest = [0] * n
        scheduled = [False] * n
        ready: list[int] = [i for i, u in enumerate(unsatisfied) if u == 0]
        scheduled_count = 0
        bundles: list[dict[str, list[tuple]]] = []
        cycle = 0

        while scheduled_count < n:
            bundle: dict[str, list[tuple]] = {}

            ready_now = [tid for tid in ready if (not scheduled[tid]) and earliest[tid] <= cycle]
            if not ready_now:
                bundles.append({"alu": [("+", dummy_dest, dummy_dest, dummy_dest)]})
                cycle += 1
                continue

            by_engine: dict[str, list[int]] = {e: [] for e in engine_order}
            for tid in ready_now:
                by_engine[tasks[tid].engine].append(tid)

            for engine in engine_order:
                cap = SLOT_LIMITS[engine]
                cands = by_engine.get(engine, [])
                if not cands:
                    continue
                # Heuristic: prefer shorter critical-path tasks (may improve overlap).
                cands.sort(key=lambda tid: (tasks[tid].cp, tid))
                take = cands[:cap]
                if not take:
                    continue
                bundle[engine] = [tasks[tid].slot for tid in take]
                for tid in take:
                    scheduled[tid] = True
                    scheduled_count += 1
                    for succ in tasks[tid].succs:
                        unsatisfied[succ] -= 1
                        earliest[succ] = max(earliest[succ], cycle + 1)
                        if unsatisfied[succ] == 0:
                            ready.append(succ)

            if not bundle:
                bundles.append({"alu": [("+", dummy_dest, dummy_dest, dummy_dest)]})
            else:
                bundles.append(bundle)
            cycle += 1

        return bundles

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel for the provided problem size.

        Key ideas:
        - Keep `values` and `indices` in scratch across rounds (avoid per-round
          loads/stores of input arrays).
        - Use SIMD `valu` ops heavily (including `multiply_add` to fuse three of
          the hash stages).
        - Use a simple VLIW list scheduler to pack independent ops into the
          same cycle, overlapping load/compute where possible.
        """
        assert batch_size % VLEN == 0, "This optimized kernel assumes full vectors"
        n_groups = batch_size // VLEN

        # Reset any pre-existing program if someone reuses the instance.
        self.instrs = []

        # Memory layout is fixed by build_mem_image():
        header = 7
        forest_values_p = header
        inp_values_p = header + n_nodes + batch_size

        tasks: list[KernelBuilder._Task] = []
        last_writer: dict[int, int] = {}
        last_reader: dict[int, int] = {}

        # -------- Scratch allocation --------
        # Per-element vectors, contiguous so each group is a vector.
        vals_base = self.alloc_scratch("vals", length=batch_size)
        idxs_base = self.alloc_scratch("idxs", length=batch_size)
        tmps_base = self.alloc_scratch("tmps", length=batch_size)
        addrs_base = self.alloc_scratch("addrs", length=batch_size)
        nodes_base = self.alloc_scratch("nodes", length=batch_size)

        # Scalar addresses for vload/vstore of the input values.
        val_ptrs = [self.alloc_scratch(f"val_ptr_{gi}") for gi in range(n_groups)]

        # -------- Constants (scalars + vectors) --------
        def const_scalar(v: int, name: str) -> int:
            addr = self.alloc_scratch(name)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="load",
                slot=("const", addr, v),
                reads=(),
                writes=(addr,),
            )
            return addr

        def vbroadcast(src_scalar: int, name: str) -> int:
            vec = self.alloc_scratch(name, length=VLEN)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("vbroadcast", vec, src_scalar),
                reads=(src_scalar,),
                writes=self._vec_addrs(vec),
            )
            return vec

        c_one = const_scalar(1, "c_one")
        c_two = const_scalar(2, "c_two")

        v_one = vbroadcast(c_one, "v_one")
        v_two = vbroadcast(c_two, "v_two")

        # Hash constants and fused multipliers.
        c0 = const_scalar(0x7ED55D16, "hash_c0")
        c1 = const_scalar(0xC761C23C, "hash_c1")
        c2 = const_scalar(0x165667B1, "hash_c2")
        c3 = const_scalar(0xD3A2646C, "hash_c3")
        c4 = const_scalar(0xFD7046C5, "hash_c4")
        c5 = const_scalar(0xB55A4F09, "hash_c5")

        m0 = const_scalar(1 + (1 << 12), "hash_m0")  # 4097
        m2 = const_scalar(1 + (1 << 5), "hash_m2")  # 33
        m4 = const_scalar(1 + (1 << 3), "hash_m4")  # 9

        s19 = const_scalar(19, "hash_s19")
        s9 = const_scalar(9, "hash_s9")
        s16 = const_scalar(16, "hash_s16")

        v_c0 = vbroadcast(c0, "v_hash_c0")
        v_c1 = vbroadcast(c1, "v_hash_c1")
        v_c2 = vbroadcast(c2, "v_hash_c2")
        v_c3 = vbroadcast(c3, "v_hash_c3")
        v_c4 = vbroadcast(c4, "v_hash_c4")
        v_c5 = vbroadcast(c5, "v_hash_c5")

        v_m0 = vbroadcast(m0, "v_hash_m0")
        v_m2 = vbroadcast(m2, "v_hash_m2")
        v_m4 = vbroadcast(m4, "v_hash_m4")

        v_s19 = vbroadcast(s19, "v_hash_s19")
        v_s9 = vbroadcast(s9, "v_hash_s9")
        v_s16 = vbroadcast(s16, "v_hash_s16")

        # Pointer constants when carrying absolute node *addresses* in the idx vector.
        c_base_plus1 = const_scalar(forest_values_p + 1, "forest_base_plus1")
        c_one_minus_base = const_scalar(1 - forest_values_p, "one_minus_forest_base")
        v_base_plus1 = vbroadcast(c_base_plus1, "v_forest_base_plus1")
        v_one_minus_base = vbroadcast(c_one_minus_base, "v_one_minus_forest_base")

        # Top-level tree node vectors for depth 0/1/2 selection rounds.
        # Load scalars from memory then broadcast.
        def load_tree_scalar(idx: int, name: str) -> int:
            addr = const_scalar(forest_values_p + idx, f"tree_addr_{idx}")
            dst = self.alloc_scratch(name)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="load",
                slot=("load", dst, addr),
                reads=(addr,),
                writes=(dst,),
            )
            return dst

        tree0 = load_tree_scalar(0, "tree0")
        v_tree1 = vbroadcast(load_tree_scalar(1, "tree1"), "v_tree1")
        v_tree2 = vbroadcast(load_tree_scalar(2, "tree2"), "v_tree2")
        v_tree3 = vbroadcast(load_tree_scalar(3, "tree3"), "v_tree3")
        v_tree4 = vbroadcast(load_tree_scalar(4, "tree4"), "v_tree4")
        v_tree5 = vbroadcast(load_tree_scalar(5, "tree5"), "v_tree5")
        v_tree6 = vbroadcast(load_tree_scalar(6, "tree6"), "v_tree6")

        # -------- Load input values into scratch --------
        for gi in range(n_groups):
            ptr = val_ptrs[gi]
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="load",
                slot=("const", ptr, inp_values_p + gi * VLEN),
                reads=(),
                writes=(ptr,),
            )
            vdst = vals_base + gi * VLEN
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="load",
                slot=("vload", vdst, ptr),
                reads=(ptr,),
                writes=self._vec_addrs(vdst),
            )

        # -------- Helpers to emit per-group ops --------
        def emit_hash(val: int, tmp_a: int, tmp_b: int):
            # Stage 0: a = a*4097 + c0
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("multiply_add", val, val, v_m0, v_c0),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_m0), *self._vec_addrs(v_c0)),
                writes=self._vec_addrs(val),
            )

            # Stage 1: a = (a ^ c1) ^ (a >> 19)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=(">>", tmp_a, val, v_s19),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_s19)),
                writes=self._vec_addrs(tmp_a),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("^", tmp_b, val, v_c1),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_c1)),
                writes=self._vec_addrs(tmp_b),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("^", val, tmp_a, tmp_b),
                reads=(*self._vec_addrs(tmp_a), *self._vec_addrs(tmp_b)),
                writes=self._vec_addrs(val),
            )

            # Stage 2: a = a*33 + c2
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("multiply_add", val, val, v_m2, v_c2),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_m2), *self._vec_addrs(v_c2)),
                writes=self._vec_addrs(val),
            )

            # Stage 3: a = (a + c3) ^ (a << 9)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("<<", tmp_a, val, v_s9),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_s9)),
                writes=self._vec_addrs(tmp_a),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("+", tmp_b, val, v_c3),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_c3)),
                writes=self._vec_addrs(tmp_b),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("^", val, tmp_a, tmp_b),
                reads=(*self._vec_addrs(tmp_a), *self._vec_addrs(tmp_b)),
                writes=self._vec_addrs(val),
            )

            # Stage 4: a = a*9 + c4
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("multiply_add", val, val, v_m4, v_c4),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_m4), *self._vec_addrs(v_c4)),
                writes=self._vec_addrs(val),
            )

            # Stage 5: a = (a ^ c5) ^ (a >> 16)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=(">>", tmp_a, val, v_s16),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_s16)),
                writes=self._vec_addrs(tmp_a),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("^", tmp_b, val, v_c5),
                reads=(*self._vec_addrs(val), *self._vec_addrs(v_c5)),
                writes=self._vec_addrs(tmp_b),
            )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("^", val, tmp_a, tmp_b),
                reads=(*self._vec_addrs(tmp_a), *self._vec_addrs(tmp_b)),
                writes=self._vec_addrs(val),
            )

        def emit_next_ptr_depth0(ptr: int, val: int, tmp: int):
            # ptr = (forest_values_p + 1) + (val & 1)  (absolute address of node 1 or 2)
            for off in range(VLEN):
                self._mk_task(
                    tasks,
                    last_writer,
                    last_reader,
                    engine="alu",
                    slot=("&", tmp + off, val + off, c_one),
                    reads=(val + off, c_one),
                    writes=(tmp + off,),
                )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("+", ptr, tmp, v_base_plus1),
                reads=(*self._vec_addrs(tmp), *self._vec_addrs(v_base_plus1)),
                writes=self._vec_addrs(ptr),
            )

        def emit_next_ptr(ptr: int, val: int, tmp: int):
            # ptr = (ptr*2 + (1 - forest_values_p)) + (val & 1)
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("multiply_add", ptr, ptr, v_two, v_one_minus_base),
                reads=(
                    *self._vec_addrs(ptr),
                    *self._vec_addrs(v_two),
                    *self._vec_addrs(v_one_minus_base),
                ),
                writes=self._vec_addrs(ptr),
            )
            for off in range(VLEN):
                self._mk_task(
                    tasks,
                    last_writer,
                    last_reader,
                    engine="alu",
                    slot=("&", tmp + off, val + off, c_one),
                    reads=(val + off, c_one),
                    writes=(tmp + off,),
                )
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="valu",
                slot=("+", ptr, ptr, tmp),
                reads=(*self._vec_addrs(ptr), *self._vec_addrs(tmp)),
                writes=self._vec_addrs(ptr),
            )

        # -------- Main rounds (no global barriers; scheduler interleaves work) --------
        groups = [
            (
                vals_base + gi * VLEN,
                idxs_base + gi * VLEN,
                tmps_base + gi * VLEN,
                addrs_base + gi * VLEN,
                nodes_base + gi * VLEN,
            )
            for gi in range(n_groups)
        ]

        for (val, ptr, tmp, addr, node) in groups:
            for r in range(rounds):
                # Choose how to obtain node values for this round.
                if r in (0, 11):
                    # depth 0: node is always root
                    for off in range(VLEN):
                        self._mk_task(
                            tasks,
                            last_writer,
                            last_reader,
                            engine="alu",
                            slot=("^", val + off, val + off, tree0),
                            reads=(val + off, tree0),
                            writes=(val + off,),
                        )
                    emit_hash(val, tmp, node)
                    emit_next_ptr_depth0(ptr, val, tmp)
                    continue

                if r in (1, 12):
                    # depth 1: node is 1 or 2, selected by ptr parity (forest base is odd, so parity is flipped)
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="valu",
                        slot=("&", tmp, ptr, v_one),
                        reads=(*self._vec_addrs(ptr), *self._vec_addrs(v_one)),
                        writes=self._vec_addrs(tmp),
                    )
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="flow",
                        slot=("vselect", node, tmp, v_tree2, v_tree1),
                        reads=(
                            *self._vec_addrs(tmp),
                            *self._vec_addrs(v_tree2),
                            *self._vec_addrs(v_tree1),
                        ),
                        writes=self._vec_addrs(node),
                    )
                    for off in range(VLEN):
                        self._mk_task(
                            tasks,
                            last_writer,
                            last_reader,
                            engine="alu",
                            slot=("^", val + off, val + off, node + off),
                            reads=(val + off, node + off),
                            writes=(val + off,),
                        )
                    emit_hash(val, tmp, node)
                    emit_next_ptr(ptr, val, tmp)
                    continue

                if r in (2, 13):
                    # depth 2: idx in [3..6], select among 4 node values.
                    # idxp = idx + 1; where idxp == ptr + (1 - forest_values_p)
                    # b0 = idxp & 2, b1 = idxp & 1
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="valu",
                        slot=("+", addr, ptr, v_one_minus_base),
                        reads=(*self._vec_addrs(ptr), *self._vec_addrs(v_one_minus_base)),
                        writes=self._vec_addrs(addr),
                    )
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="valu",
                        slot=("&", tmp, addr, v_one),
                        reads=(*self._vec_addrs(addr), *self._vec_addrs(v_one)),
                        writes=self._vec_addrs(tmp),
                    )
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="valu",
                        slot=("&", addr, addr, v_two),
                        reads=(*self._vec_addrs(addr), *self._vec_addrs(v_two)),
                        writes=self._vec_addrs(addr),
                    )
                    # node = (b1 ? tree4 : tree3)
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="flow",
                        slot=("vselect", node, tmp, v_tree4, v_tree3),
                        reads=(
                            *self._vec_addrs(tmp),
                            *self._vec_addrs(v_tree4),
                            *self._vec_addrs(v_tree3),
                        ),
                        writes=self._vec_addrs(node),
                    )
                    # tmp = (b1 ? tree6 : tree5)
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="flow",
                        slot=("vselect", tmp, tmp, v_tree6, v_tree5),
                        reads=(
                            *self._vec_addrs(tmp),
                            *self._vec_addrs(v_tree6),
                            *self._vec_addrs(v_tree5),
                        ),
                        writes=self._vec_addrs(tmp),
                    )
                    # node = (b0 ? tmp : node)
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="flow",
                        slot=("vselect", node, addr, tmp, node),
                        reads=(
                            *self._vec_addrs(addr),
                            *self._vec_addrs(tmp),
                            *self._vec_addrs(node),
                        ),
                        writes=self._vec_addrs(node),
                    )
                    for off in range(VLEN):
                        self._mk_task(
                            tasks,
                            last_writer,
                            last_reader,
                            engine="alu",
                            slot=("^", val + off, val + off, node + off),
                            reads=(val + off, node + off),
                            writes=(val + off,),
                        )
                    emit_hash(val, tmp, node)
                    emit_next_ptr(ptr, val, tmp)
                    continue

                # Gather from memory for deeper rounds.
                for off in range(VLEN):
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="load",
                        slot=("load_offset", node, ptr, off),
                        reads=(ptr + off,),
                        writes=(node + off,),
                    )
                for off in range(VLEN):
                    self._mk_task(
                        tasks,
                        last_writer,
                        last_reader,
                        engine="alu",
                        slot=("^", val + off, val + off, node + off),
                        reads=(val + off, node + off),
                        writes=(val + off,),
                    )
                emit_hash(val, tmp, node)
                emit_next_ptr(ptr, val, tmp)

        # -------- Store output values back to memory --------
        for gi in range(n_groups):
            ptr = val_ptrs[gi]
            vsrc = vals_base + gi * VLEN
            self._mk_task(
                tasks,
                last_writer,
                last_reader,
                engine="store",
                slot=("vstore", ptr, vsrc),
                reads=(ptr, *self._vec_addrs(vsrc)),
                writes=(),
            )

        # -------- Schedule into VLIW instruction bundles --------
        self._compute_cp(tasks)
        dummy = self.alloc_scratch("sched_dummy")
        self.instrs = self._schedule(tasks, dummy_dest=dummy)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    # Match the submission harness: no pauses, no debug compares.
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect result"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
