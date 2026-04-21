const std = @import("std");
const Allocator = std.mem.Allocator;
const Deque = @import("deque.zig").Deque;

pub const WaitGroup = std.Thread.WaitGroup;

pub const Error = error{
    InvalidThreadCount,
    ThreadCountTooLarge,
} || Allocator.Error || std.Thread.SpawnError;

/// Work-stealing thread pool with Rayon-style `join` and `scope` primitives.
///
/// Default thread count is `getCpuCount() / 2` capped to `max_thread_count`,
/// leaving cores for I/O, event loops, etc. Tasks are distributed via
/// per-worker Chase-Lev deques with LIFO local execution and FIFO stealing for
/// cache-friendly load balancing.
///
/// Lifetime contract:
///   - All tasks spawned via `spawnWg`, `Scope.spawn`, or `join` MUST complete before
///     the pool is deinitialized. Callers enforce this via `waitAndWork`, `scope`, or
///     by finishing a `join`. `deinit` panics if tasks are still pending.
///   - `deinit` must be called exactly once. Calling it twice panics.
///
/// Thread-affinity:
///   - `ThreadPool.*` entry points (`spawnWg`, `scope`, `join`, `waitAndWork`) are
///     safe to call from any thread. Once `deinit` begins, new entries panic and
///     deinit waits for in-flight ThreadPool calls to leave before freeing memory.
///   - A `Scope*` may be shared across threads; only `pending` is touched atomically.
///     Do not mutate Scope fields from user code — treat the pointer as read-only.
///   - `join` gives up parallelism gracefully: if called from outside a worker it still
///     forks B to the global queue, never falling back to sequential execution.
///
/// Allocator contract:
///   - `Config.allocator` is invoked from arbitrary threads (worker threads for
///     task wrappers and deque growth, caller threads for `spawnWg`/`scope.spawn`).
///     It MUST be thread-safe. `std.heap.GeneralPurposeAllocator(.{})`, the C
///     allocator, and the page allocator all qualify; `ArenaAllocator` does not.
///
/// Panic contract (IMPORTANT):
///   - User functions passed to `spawnWg`, `Scope.spawn`, `join`, and `scope` MUST
///     NOT panic. Zig panics do not unwind `defer` blocks; a panicking user
///     function will:
///       * tear down a worker thread (silently reducing pool throughput, and
///         eventually deadlocking waiters if enough workers die),
///       * skip `scopeWait` / the `waitForJoin` defer, leaving in-flight tasks
///         holding pointers into a torn-down caller stack frame (UAF on the
///         surviving threads between the panic and process abort).
///     Return `error!void` for fallible work and propagate errors via `scope`'s
///     error-union signature.
pub const ThreadPool = struct {
    const Self = @This();

    workers: []Worker,
    threads: []std.Thread,
    shutdown: std.atomic.Value(bool),
    /// High bit closes the pool to new ThreadPool entry points during deinit;
    /// low bits count in-flight ThreadPool calls that must drain before free.
    call_state: std.atomic.Value(u64),
    global_queue: GlobalQueue,
    park_mutex: std.Thread.Mutex,
    park_cond: std.Thread.Condition,
    lifecycle_mutex: std.Thread.Mutex,
    lifecycle_cond: std.Thread.Condition,
    /// Monotonically increasing counter bumped on every schedule. Workers snapshot
    /// this before scanning for work and compare under park_mutex before sleeping,
    /// ensuring no signal is missed between "no work found" and wait().
    work_counter: std.atomic.Value(u64),
    /// Number of threads currently parked on park_cond. Notifiers skip the
    /// lock+broadcast when zero. Correctness rests on a Dekker-style proof:
    /// waiters inc this BEFORE checking their predicate, notifiers mutate the
    /// predicate BEFORE reading this, both with seq_cst — any execution where
    /// a waiter parks AND a notifier skips would close a four-event cycle.
    parked_count: std.atomic.Value(u32),
    /// Effectively immutable after `init`. The field is left non-const only
    /// to support failure-injection tests; production code must not mutate.
    /// Invoked from arbitrary threads — must be thread-safe (see doc comment).
    allocator: Allocator,
    thread_count: u32,

    /// Upper bound on caller-supplied thread count. Guards against resource
    /// exhaustion when `Config.thread_count` is influenced by untrusted input.
    pub const max_thread_count: u32 = 1024;

    /// Max time the park loop sleeps before re-checking its wait condition.
    /// Bounds worst-case wakeup latency when a completion path doesn't broadcast.
    const park_timeout_ns: u64 = 10 * std.time.ns_per_ms;
    const call_state_closed_mask: u64 = @as(u64, 1) << 63;
    const call_state_count_mask: u64 = call_state_closed_mask - 1;

    const GlobalQueue = struct {
        mutex: std.Thread.Mutex = .{},
        head: ?*Task = null,
        tail: ?*Task = null,

        fn push(self: *GlobalQueue, task: *Task) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            task.next = null;
            if (self.tail) |tail| {
                tail.next = task;
            } else {
                self.head = task;
            }
            self.tail = task;
        }

        fn pop(self: *GlobalQueue) ?*Task {
            self.mutex.lock();
            defer self.mutex.unlock();
            const task = self.head orelse return null;
            self.head = task.next;
            if (self.head == null) {
                self.tail = null;
            }
            task.next = null;
            return task;
        }

        fn isEmpty(self: *GlobalQueue) bool {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.head == null;
        }

        /// O(n). Diagnostic only — used by `deinit`'s pending-tasks panic so
        /// the operator sees the actual leftover count rather than a boolean.
        fn len(self: *GlobalQueue) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            var n: usize = 0;
            var cur = self.head;
            while (cur) |t| : (cur = t.next) n += 1;
            return n;
        }
    };

    pub const Config = struct {
        allocator: Allocator,
        thread_count: ?u32 = null,
    };

    /// Create a new thread pool. The returned pointer is heap-allocated and stable
    /// (workers hold back-pointers to it).
    ///
    /// Returns `error.InvalidThreadCount` if `thread_count == 0` and
    /// `error.ThreadCountTooLarge` if it exceeds `max_thread_count`.
    pub fn init(config: Config) Error!*Self {
        if (config.thread_count) |tc| {
            if (tc == 0) return error.InvalidThreadCount;
            if (tc > max_thread_count) return error.ThreadCountTooLarge;
        }
        const cpu_count = std.Thread.getCpuCount() catch 2;
        // Cap auto-derived thread count against max_thread_count so absurd
        // cpu_count values (e.g. bogus cgroup reporting in containers) cannot
        // bypass the explicit limit or overflow @intCast(u32).
        const auto_count: usize = @min(@as(usize, max_thread_count), @max(@as(usize, 1), cpu_count / 2));
        const thread_count: u32 = config.thread_count orelse @intCast(auto_count);

        const workers = try config.allocator.alloc(Worker, thread_count);
        var deques_initialized: u32 = 0;
        var owns_worker_cleanup = true;
        errdefer {
            if (owns_worker_cleanup) {
                for (workers[0..deques_initialized]) |*w| {
                    w.deque.deinit();
                }
                config.allocator.free(workers);
            }
        }

        for (workers, 0..) |*w, i| {
            w.* = .{
                .deque = try Deque(*Task).init(config.allocator),
                .pool = undefined,
                .index = @intCast(i),
            };
            deques_initialized += 1;
        }

        const threads = try config.allocator.alloc(std.Thread, thread_count);
        errdefer config.allocator.free(threads);

        const pool = try config.allocator.create(Self);
        errdefer config.allocator.destroy(pool);

        pool.* = .{
            .workers = workers,
            .threads = threads,
            .shutdown = std.atomic.Value(bool).init(false),
            .call_state = std.atomic.Value(u64).init(0),
            .global_queue = .{},
            .park_mutex = .{},
            .park_cond = .{},
            .lifecycle_mutex = .{},
            .lifecycle_cond = .{},
            .work_counter = std.atomic.Value(u64).init(0),
            .parked_count = std.atomic.Value(u32).init(0),
            .allocator = config.allocator,
            .thread_count = thread_count,
        };

        for (pool.workers) |*w| {
            w.pool = pool;
        }

        // From here, the spawn errdefer owns worker cleanup
        owns_worker_cleanup = false;
        var spawned: u32 = 0;
        errdefer {
            pool.shutdown.store(true, .release);
            pool.park_mutex.lock();
            pool.park_cond.broadcast();
            pool.park_mutex.unlock();
            for (0..spawned) |j| {
                pool.threads[j].join();
            }
            for (pool.workers) |*w| {
                w.deque.deinit();
            }
            config.allocator.free(workers);
        }
        for (pool.threads, pool.workers) |*t, *w| {
            t.* = try std.Thread.spawn(.{}, workerLoop, .{w});
            spawned += 1;
            // Best-effort: `bufPrint` can overflow on platforms where
            // max_name_len is small (e.g. 16 on Linux, which cannot fit
            // "pool-worker-1024\x00"), and setName can fail for platform
            // reasons. Both failures are benign — threads run regardless.
            var name_buf: [std.Thread.max_name_len]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "pool-worker-{d}", .{w.index}) catch continue;
            t.setName(name) catch {};
        }

        return pool;
    }

    /// Shut down the pool and free resources.
    ///
    /// Preconditions (enforced by panic):
    ///   - All tasks spawned on this pool have completed (caller drained their
    ///     WaitGroup / Scope / join before calling). Leftover tasks panic rather
    ///     than run post-shutdown — running user callbacks on the teardown thread
    ///     would re-enter caller state that may already be torn down.
    ///   - `deinit` is called exactly once.
    pub fn deinit(self: *Self) void {
        const previous_state = self.call_state.fetchOr(call_state_closed_mask, .acq_rel);
        if ((previous_state & call_state_closed_mask) != 0) {
            @panic("ThreadPool.deinit called more than once");
        }

        self.lifecycle_mutex.lock();
        while ((self.call_state.load(.acquire) & call_state_count_mask) != 0) {
            self.lifecycle_cond.wait(&self.lifecycle_mutex);
        }
        self.lifecycle_mutex.unlock();

        self.shutdown.store(true, .release);
        self.park_mutex.lock();
        self.park_cond.broadcast();
        self.park_mutex.unlock();
        for (self.threads) |t| {
            t.join();
        }

        // Fail closed: if the caller forgot to wait, surface it rather than
        // silently executing orphaned task callbacks on this thread. Leaked
        // worker/thread/wrapper allocations on the panic path are acceptable
        // because the process is about to abort.
        var deque_pending: usize = 0;
        for (self.workers) |*w| {
            // Post-join the deque is quiescent. A wraparound here (size >
            // capacity) would indicate deque corruption — surface it rather
            // than masking via count()'s defensive zero-return.
            const b = w.deque.bottom.load(.acquire);
            const t = w.deque.top.load(.acquire);
            const raw_size = b -% t;
            const cap = w.deque.buffer.load(.acquire).mask + 1;
            if (raw_size > cap) {
                @panic("ThreadPool.deinit observed a corrupted worker deque (bottom < top)");
            }
            deque_pending += raw_size;
        }
        const global_pending = self.global_queue.len();
        if (deque_pending + global_pending > 0) {
            @panic("ThreadPool.deinit called with pending tasks — wait for all spawned work before deinit");
        }

        for (self.workers) |*w| {
            w.deque.deinit();
        }
        const allocator = self.allocator;
        allocator.free(self.workers);
        allocator.free(self.threads);
        allocator.destroy(self);
    }

    // =========================================================================
    // Task
    // =========================================================================

    pub const Task = struct {
        callback: *const fn (*Task) void,
        next: ?*Task = null,
    };

    const CallGuard = struct {
        pool: *Self,

        fn leave(self: *CallGuard) void {
            self.pool.leaveCall();
        }
    };

    // =========================================================================
    // spawnWg / waitAndWork
    // =========================================================================

    /// Spawn a task whose completion is tracked by `wg`. Returns `error.OutOfMemory`
    /// if the task wrapper cannot be allocated — callers must handle this rather
    /// than receiving a silently-dropped task.
    pub fn spawnWg(
        self: *Self,
        wg: *WaitGroup,
        comptime func: anytype,
        args: anytype,
    ) Allocator.Error!void {
        // Guard against schedule's signature drifting — see schedule's docs.
        comptime std.debug.assert(@typeInfo(@TypeOf(schedule)).@"fn".return_type.? == void);
        var call = self.enterCall();
        defer call.leave();

        const Args = @TypeOf(args);
        const Wrapper = struct {
            args: Args,
            pool: *Self,
            wg: *WaitGroup,
            task: Task = .{ .callback = run },

            fn run(task: *Task) void {
                const wrapper: *@This() = @fieldParentPtr("task", task);
                const alloc = wrapper.pool.allocator;
                const pool = wrapper.pool;
                const wg_ptr = wrapper.wg;
                @call(.auto, func, wrapper.args);
                alloc.destroy(wrapper);
                wg_ptr.finish();
                // Unconditionally broadcast rather than using the parked_count
                // gate: WaitGroup's internal ordering isn't seq_cst, so the
                // Dekker pair (inc parked_count / load isDone) doesn't close a
                // cycle with finish's release. Hot under high spawn rates —
                // if this becomes a bottleneck, strengthen WaitGroup's ordering
                // and switch to maybeWakeWaiters.
                pool.park_mutex.lock();
                pool.park_cond.broadcast();
                pool.park_mutex.unlock();
            }
        };

        const wrapper = try self.allocator.create(Wrapper);
        wrapper.* = .{ .args = args, .pool = self, .wg = wg };
        wg.start();
        self.schedule(&wrapper.task);
    }

    /// Block until `wg` drains, processing pool tasks while waiting.
    /// Safe to call from any thread.
    pub fn waitAndWork(self: *Self, wg: *WaitGroup) void {
        var call = self.enterCall();
        defer call.leave();

        var idle_spins: u32 = 0;
        while (!wg.isDone()) {
            if (self.findAndRunTask()) {
                idle_spins = 0;
                continue;
            }
            idle_spins += 1;
            if (idle_spins < 32) {
                std.atomic.spinLoopHint();
                continue;
            }
            // Park with a short timeout: completions broadcast, and the timeout
            // bounds latency for waiters that race with a park transition.
            const snapshot = self.work_counter.load(.acquire);
            self.park_mutex.lock();
            // Inc parked_count BEFORE checking the predicate (see parked_count
            // docstring for the proof). wg's internal ordering is acquire/release
            // not seq_cst, so the parked_count gate doesn't apply to wg.finish's
            // broadcast — but wg's wrapper.run still always broadcasts.
            _ = self.parked_count.fetchAdd(1, .seq_cst);
            if (!wg.isDone() and self.work_counter.load(.seq_cst) == snapshot) {
                self.park_cond.timedWait(&self.park_mutex, park_timeout_ns) catch {};
            }
            _ = self.parked_count.fetchSub(1, .seq_cst);
            self.park_mutex.unlock();
            idle_spins = 0;
        }
    }

    // =========================================================================
    // scope
    // =========================================================================

    /// Structured concurrency handle. All tasks spawned on a scope must complete
    /// before the enclosing `scope()` call returns.
    ///
    /// The pointer may be shared with spawned tasks (including nested spawns),
    /// but user code must treat fields as read-only — only `pending` is atomic.
    pub const Scope = struct {
        pool: *Self,
        // u64 so that wraparound on unbounded spawn patterns is effectively
        // impossible (2^64 sub-tasks would run for centuries); u32 could in
        // principle silently wrap and trip `scopeWait`'s exit condition early.
        pending: std.atomic.Value(u64),
        allocator: Allocator,

        /// Spawn a task inside this scope. Returns `error.OutOfMemory` if the
        /// wrapper cannot be allocated; the scope's `pending` count is unchanged
        /// on error so the enclosing `scope()` is not left hanging.
        pub fn spawn(
            self: *Scope,
            comptime func: anytype,
            args: anytype,
        ) Allocator.Error!void {
            // Guard against schedule's signature drifting — see schedule's docs.
            comptime std.debug.assert(@typeInfo(@TypeOf(schedule)).@"fn".return_type.? == void);
            const Args = @TypeOf(args);
            const Wrapper = struct {
                args: Args,
                scope: *Scope,
                task: Task = .{ .callback = run },

                fn run(task: *Task) void {
                    const wrapper: *@This() = @fieldParentPtr("task", task);
                    // Capture everything off `scope` BEFORE the fetchSub that
                    // may release the enclosing `scope()` frame — after that
                    // point, `wrapper.scope` is no longer safe to dereference.
                    const alloc = wrapper.scope.allocator;
                    const pool = wrapper.scope.pool;
                    const pending_ptr = &wrapper.scope.pending;
                    @call(.auto, func, wrapper.args);
                    alloc.destroy(wrapper);
                    // seq_cst on the fetchSub pairs with scopeWait's seq_cst load
                    // and parked_count check (see parked_count docstring).
                    const prev = pending_ptr.fetchSub(1, .seq_cst);
                    if (prev == 1) {
                        pool.maybeWakeWaiters();
                    }
                }
            };

            const wrapper = try self.allocator.create(Wrapper);
            wrapper.* = .{ .args = args, .scope = self };
            _ = self.pending.fetchAdd(1, .acq_rel);
            self.pool.schedule(&wrapper.task);
        }
    };

    /// Run `func(&scope, args...)` and block until every task spawned on `scope`
    /// has completed. Propagates errors from `func`.
    ///
    /// `func` MUST surface failures via the error union — a panic from `func`
    /// skips the deferred wait and leaves spawned tasks holding a pointer to a
    /// torn-down `Scope` on this thread's stack (use-after-free).
    pub fn scope(self: *Self, comptime func: anytype, args: anytype) CallReturn(@TypeOf(func)) {
        var call = self.enterCall();
        defer call.leave();

        var s = Scope{
            .pool = self,
            .pending = std.atomic.Value(u64).init(0),
            .allocator = self.allocator,
        };

        const R = @TypeOf(@call(.auto, func, .{&s} ++ args));
        defer self.scopeWait(&s);

        if (@typeInfo(R) == .error_union) {
            try @call(.auto, func, .{&s} ++ args);
        } else {
            @call(.auto, func, .{&s} ++ args);
        }
    }

    fn scopeWait(self: *Self, s: *Scope) void {
        var idle_spins: u32 = 0;
        while (s.pending.load(.seq_cst) > 0) {
            if (self.findAndRunTask()) {
                idle_spins = 0;
                continue;
            }
            idle_spins += 1;
            if (idle_spins < 32) {
                std.atomic.spinLoopHint();
                continue;
            }
            const snapshot = self.work_counter.load(.acquire);
            self.park_mutex.lock();
            _ = self.parked_count.fetchAdd(1, .seq_cst);
            if (s.pending.load(.seq_cst) > 0 and self.work_counter.load(.seq_cst) == snapshot) {
                self.park_cond.timedWait(&self.park_mutex, park_timeout_ns) catch {};
            }
            _ = self.parked_count.fetchSub(1, .seq_cst);
            self.park_mutex.unlock();
            idle_spins = 0;
        }
    }

    fn CallReturn(comptime Func: type) type {
        const info = @typeInfo(Func);
        const fn_info = switch (info) {
            .@"fn" => |f| f,
            .pointer => |p| @typeInfo(p.child).@"fn",
            else => @compileError("scope requires a function"),
        };
        const R = fn_info.return_type orelse void;
        return switch (@typeInfo(R)) {
            .void => void,
            .error_union => |eu| switch (@typeInfo(eu.payload)) {
                .void => eu.error_set!void,
                else => @compileError("scope user function must return void or !void; got " ++ @typeName(R)),
            },
            else => @compileError("scope user function must return void or !void; got " ++ @typeName(R)),
        };
    }

    // =========================================================================
    // join
    // =========================================================================

    /// Run `a_fn(a_args...)` and `b_fn(b_args...)` in parallel (when possible),
    /// returning both results. If local work-stealing resources are saturated,
    /// B is published to the global queue rather than executed sequentially —
    /// the API's concurrency contract holds regardless of deque pressure.
    ///
    /// Neither `a_fn` nor `b_fn` may panic: the `JoinTask` lives on this
    /// thread's stack and a panic skips the wait, leaving an in-flight
    /// stealer-run B with a dangling pointer.
    pub fn join(
        self: *Self,
        comptime a_fn: anytype,
        a_args: anytype,
        comptime b_fn: anytype,
        b_args: anytype,
    ) JoinResult(@TypeOf(@call(.auto, a_fn, a_args)), @TypeOf(@call(.auto, b_fn, b_args))) {
        var call = self.enterCall();
        defer call.leave();

        const BArgs = @TypeOf(b_args);

        const JoinTask = struct {
            b_args: BArgs,
            result: @TypeOf(@call(.auto, b_fn, b_args)) = undefined,
            done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
            pool: *Self,
            task: Task = .{ .callback = run },

            fn run(task: *Task) void {
                const jt: *@This() = @fieldParentPtr("task", task);
                const pool = jt.pool;
                jt.result = @call(.auto, b_fn, jt.b_args);
                // seq_cst on done pairs with waitForJoin's seq_cst load and
                // parked_count check (see parked_count docstring).
                jt.done.store(true, .seq_cst);
                pool.maybeWakeWaiters();
            }
        };

        var jt = JoinTask{ .b_args = b_args, .pool = self };

        if (self.localWorker()) |worker| {
            var reclaimable = true;
            worker.deque.push(&jt.task) catch {
                // Local deque full — publish globally so a worker can pick B up.
                // We can no longer reclaim from the owner deque, but fork-join
                // parallelism is preserved.
                self.global_queue.push(&jt.task);
                reclaimable = false;
            };
            self.notifyWorker();

            const a_result = @call(.auto, a_fn, a_args);

            if (reclaimable) {
                if (worker.deque.pop()) |popped_task| {
                    if (popped_task == &jt.task) {
                        const b_result = @call(.auto, b_fn, b_args);
                        return .{ .a = a_result, .b = b_result };
                    } else {
                        // Not our task — a_fn pushed nested work (e.g. a
                        // nested join) on top of jt. Run it, then fall through
                        // to waitForJoin, which will drain the rest of the
                        // deque including jt via findAndRunTask.
                        popped_task.callback(popped_task);
                    }
                }
            }

            self.waitForJoin(&jt.done);
            return .{ .a = a_result, .b = jt.result };
        } else {
            // Caller isn't a worker. Push B globally so workers execute it in
            // parallel with A on the caller's thread — never fall back to
            // sequential; the API promises parallelism.
            self.global_queue.push(&jt.task);
            self.notifyWorker();

            const a_result = @call(.auto, a_fn, a_args);

            self.waitForJoin(&jt.done);
            return .{ .a = a_result, .b = jt.result };
        }
    }

    fn waitForJoin(self: *Self, done: *std.atomic.Value(bool)) void {
        var idle_spins: u32 = 0;
        while (!done.load(.seq_cst)) {
            if (self.findAndRunTask()) {
                idle_spins = 0;
                continue;
            }
            idle_spins += 1;
            if (idle_spins < 32) {
                std.atomic.spinLoopHint();
                continue;
            }
            const snapshot = self.work_counter.load(.acquire);
            self.park_mutex.lock();
            _ = self.parked_count.fetchAdd(1, .seq_cst);
            if (!done.load(.seq_cst) and self.work_counter.load(.seq_cst) == snapshot) {
                self.park_cond.timedWait(&self.park_mutex, park_timeout_ns) catch {};
            }
            _ = self.parked_count.fetchSub(1, .seq_cst);
            self.park_mutex.unlock();
            idle_spins = 0;
        }
    }

    fn JoinResult(comptime A: type, comptime B: type) type {
        return struct {
            a: A,
            b: B,
        };
    }

    // =========================================================================
    // Internal scheduling
    // =========================================================================

    /// Returns the current worker if it belongs to this pool, else null.
    fn localWorker(self: *Self) ?*Worker {
        const worker = current_worker orelse return null;
        return if (worker.pool == self) worker else null;
    }

    fn enterCall(self: *Self) CallGuard {
        while (true) {
            const state = self.call_state.load(.acquire);
            if ((state & call_state_closed_mask) != 0) {
                @panic("ThreadPool API called after deinit began");
            }
            if ((state & call_state_count_mask) == call_state_count_mask) {
                @panic("ThreadPool active call counter overflowed");
            }
            if (self.call_state.cmpxchgWeak(state, state + 1, .acq_rel, .acquire) == null) {
                return .{ .pool = self };
            }
        }
    }

    fn leaveCall(self: *Self) void {
        const previous_state = self.call_state.fetchSub(1, .acq_rel);
        std.debug.assert((previous_state & call_state_count_mask) > 0);
        if ((previous_state & call_state_closed_mask) != 0 and
            (previous_state & call_state_count_mask) == 1)
        {
            self.lifecycle_mutex.lock();
            self.lifecycle_cond.broadcast();
            self.lifecycle_mutex.unlock();
        }
    }

    /// Enqueue a task. INFALLIBLE — deque push errors (OOM on buffer grow,
    /// BufferAtMaxSize) are absorbed by falling back to the unbounded global
    /// queue. This is load-bearing for the API: `Scope.spawn` and `spawnWg`
    /// expose `Allocator.Error!void`, which would be a lie if this function
    /// could surface `error.BufferAtMaxSize`. Do NOT change schedule's
    /// signature without also tightening those callers' error sets.
    fn schedule(self: *Self, task: *Task) void {
        if (self.shutdown.load(.acquire)) {
            @panic("ThreadPool task scheduled after shutdown began");
        }
        if (self.localWorker()) |worker| {
            worker.deque.push(task) catch {
                self.global_queue.push(task);
                self.notifyWorker();
                return;
            };
            self.notifyWorker();
            return;
        }
        self.global_queue.push(task);
        self.notifyWorker();
    }

    fn notifyWorker(self: *Self) void {
        _ = self.work_counter.fetchAdd(1, .seq_cst);
        self.maybeWakeWaiters();
    }

    /// Broadcast on park_cond only when at least one thread is parked. The
    /// proof requires seq_cst on parked_count and on the predicate every
    /// caller pairs us with (work_counter for notifyWorker, jt.done for
    /// JoinTask.run, scope.pending for Scope wrapper).
    fn maybeWakeWaiters(self: *Self) void {
        if (self.parked_count.load(.seq_cst) > 0) {
            self.park_mutex.lock();
            self.park_cond.broadcast();
            self.park_mutex.unlock();
        }
    }

    fn findAndRunTask(self: *Self) bool {
        const local_worker = self.localWorker();
        if (local_worker) |worker| {
            if (worker.deque.pop()) |task| {
                task.callback(task);
                return true;
            }
        }
        if (self.global_queue.pop()) |task| {
            task.callback(task);
            return true;
        }
        if (local_worker) |worker| {
            const count = self.workers.len;
            if (count > 1) {
                const start = (@as(usize, worker.index) + 1) % count;
                if (self.tryStealing(start, @as(usize, worker.index))) |task| {
                    task.callback(task);
                    return true;
                }
            }
        } else {
            if (self.tryStealing(0, null)) |task| {
                task.callback(task);
                return true;
            }
        }
        return false;
    }

    fn tryStealing(self: *Self, start: usize, skip_index: ?usize) ?*Task {
        const count = self.workers.len;
        if (count == 0) return null;

        for (0..count) |offset| {
            const target_idx = (start + offset) % count;
            if (skip_index) |skip| {
                if (target_idx == skip) continue;
            }
            const target = &self.workers[target_idx];
            switch (target.deque.steal()) {
                .success => |task| return task,
                .empty, .retry => continue,
            }
        }
        return null;
    }

    // =========================================================================
    // Worker
    // =========================================================================

    const Worker = struct {
        deque: Deque(*Task),
        pool: *Self,
        index: u32,
    };

    threadlocal var current_worker: ?*Worker = null;

    fn workerLoop(worker: *Worker) void {
        current_worker = worker;
        defer current_worker = null;

        const pool = worker.pool;
        var idle_spins: u32 = 0;

        while (true) {
            if (pool.shutdown.load(.acquire)) return;

            // Snapshot work_counter before scanning — if it changes by the time
            // we consider parking, new work was posted and we must re-scan.
            const snapshot = pool.work_counter.load(.acquire);

            if (worker.deque.pop()) |task| {
                idle_spins = 0;
                task.callback(task);
                continue;
            }

            if (pool.global_queue.pop()) |task| {
                idle_spins = 0;
                task.callback(task);
                continue;
            }

            const count = pool.workers.len;
            if (count > 1) {
                const start = (@as(usize, worker.index) + 1) % count;
                if (pool.tryStealing(start, @as(usize, worker.index))) |task| {
                    idle_spins = 0;
                    task.callback(task);
                    continue;
                }
            }

            idle_spins += 1;
            if (idle_spins < 64) {
                std.atomic.spinLoopHint();
            } else {
                pool.park_mutex.lock();
                if (pool.shutdown.load(.acquire)) {
                    pool.park_mutex.unlock();
                    return;
                }
                // Inc parked_count BEFORE the work_counter check so that any
                // notifier that mutates work_counter after our load observes
                // parked_count > 0 and broadcasts. See parked_count docstring.
                _ = pool.parked_count.fetchAdd(1, .seq_cst);
                if (pool.work_counter.load(.seq_cst) == snapshot) {
                    pool.park_cond.timedWait(&pool.park_mutex, park_timeout_ns) catch {};
                }
                _ = pool.parked_count.fetchSub(1, .seq_cst);
                pool.park_mutex.unlock();
                idle_spins = 0;
            }
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "pool init and deinit" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    try std.testing.expectEqual(@as(u32, 2), pool.thread_count);
}

test "init rejects zero thread count" {
    try std.testing.expectError(error.InvalidThreadCount, ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 0,
    }));
}

test "init rejects oversized thread count" {
    try std.testing.expectError(error.ThreadCountTooLarge, ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = ThreadPool.max_thread_count + 1,
    }));
}

test "spawnWg basic" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    var counter = std.atomic.Value(u64).init(0);
    var wg = WaitGroup{};

    for (0..100) |_| {
        try pool.spawnWg(&wg, struct {
            fn work(c: *std.atomic.Value(u64)) void {
                _ = c.fetchAdd(1, .acq_rel);
            }
        }.work, .{&counter});
    }

    pool.waitAndWork(&wg);
    try std.testing.expectEqual(@as(u64, 100), counter.load(.acquire));
}

test "scope basic" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 4,
    });
    defer pool.deinit();

    var counter = std.atomic.Value(u64).init(0);

    try pool.scope(struct {
        fn run(s: *ThreadPool.Scope, c: *std.atomic.Value(u64)) !void {
            for (0..200) |_| {
                try s.spawn(struct {
                    fn work(cc: *std.atomic.Value(u64)) void {
                        _ = cc.fetchAdd(1, .acq_rel);
                    }
                }.work, .{c});
            }
        }
    }.run, .{&counter});

    try std.testing.expectEqual(@as(u64, 200), counter.load(.acquire));
}

test "scope propagates user function errors" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    const Err = error{Oops};
    const result = pool.scope(struct {
        fn run(_: *ThreadPool.Scope) Err!void {
            return error.Oops;
        }
    }.run, .{});
    try std.testing.expectError(Err.Oops, result);
}

test "join basic" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    var result_a = std.atomic.Value(u64).init(0);
    var result_b = std.atomic.Value(u64).init(0);

    var wg = WaitGroup{};
    try pool.spawnWg(&wg, struct {
        fn work(p: *ThreadPool, ra: *std.atomic.Value(u64), rb: *std.atomic.Value(u64)) void {
            const result = p.join(
                struct {
                    fn a() u64 {
                        return 42;
                    }
                }.a,
                .{},
                struct {
                    fn b() u64 {
                        return 99;
                    }
                }.b,
                .{},
            );
            ra.store(result.a, .release);
            rb.store(result.b, .release);
        }
    }.work, .{ pool, &result_a, &result_b });

    pool.waitAndWork(&wg);
    try std.testing.expectEqual(@as(u64, 42), result_a.load(.acquire));
    try std.testing.expectEqual(@as(u64, 99), result_b.load(.acquire));
}

test "join from non-worker forks via global queue" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    const result = pool.join(
        struct {
            fn a() u64 {
                return 1;
            }
        }.a,
        .{},
        struct {
            fn b() u64 {
                return 2;
            }
        }.b,
        .{},
    );

    try std.testing.expectEqual(@as(u64, 1), result.a);
    try std.testing.expectEqual(@as(u64, 2), result.b);
}

test "scope stress test" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 4,
    });
    defer pool.deinit();

    var sum = std.atomic.Value(u64).init(0);

    try pool.scope(struct {
        fn run(s: *ThreadPool.Scope, c: *std.atomic.Value(u64)) !void {
            for (0..10_000) |_| {
                try s.spawn(struct {
                    fn work(cc: *std.atomic.Value(u64)) void {
                        _ = cc.fetchAdd(1, .acq_rel);
                    }
                }.work, .{c});
            }
        }
    }.run, .{&sum});

    try std.testing.expectEqual(@as(u64, 10_000), sum.load(.acquire));
}

test "spawnWg propagates allocator errors" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });
    defer pool.deinit();

    // Failing allocator: every allocation returns OutOfMemory.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    // Point the pool at the failing allocator just for the spawn attempt.
    const original = pool.allocator;
    pool.allocator = failing.allocator();
    defer pool.allocator = original;

    var wg = WaitGroup{};
    const result = pool.spawnWg(&wg, struct {
        fn work() void {}
    }.work, .{});
    try std.testing.expectError(error.OutOfMemory, result);
    // wg must not have been incremented on failure.
    try std.testing.expect(wg.isDone());
}

test "waitAndWork steals worker-local tasks for external callers" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 1,
    });
    defer pool.deinit();

    var outer_wg = WaitGroup{};
    var inner_wg = WaitGroup{};
    var worker_ready = std.atomic.Value(bool).init(false);
    var release_worker = std.atomic.Value(bool).init(false);
    var child_ran = std.atomic.Value(bool).init(false);

    try pool.spawnWg(&outer_wg, struct {
        fn run(
            p: *ThreadPool,
            inner: *WaitGroup,
            ready: *std.atomic.Value(bool),
            release: *std.atomic.Value(bool),
            child_flag: *std.atomic.Value(bool),
        ) void {
            p.spawnWg(inner, struct {
                fn child(flag: *std.atomic.Value(bool)) void {
                    flag.store(true, .release);
                }
            }.child, .{child_flag}) catch unreachable;
            ready.store(true, .release);
            while (!release.load(.acquire)) {
                std.atomic.spinLoopHint();
            }
        }
    }.run, .{ pool, &inner_wg, &worker_ready, &release_worker, &child_ran });

    while (!worker_ready.load(.acquire)) {
        std.atomic.spinLoopHint();
    }

    const waiter = try std.Thread.spawn(.{}, struct {
        fn run(p: *ThreadPool, inner: *WaitGroup) void {
            p.waitAndWork(inner);
        }
    }.run, .{ pool, &inner_wg });
    defer waiter.join();
    defer pool.waitAndWork(&outer_wg);
    defer release_worker.store(true, .release);

    var ran_before_release = false;
    for (0..100) |_| {
        if (child_ran.load(.acquire)) {
            ran_before_release = true;
            break;
        }
        std.Thread.sleep(1 * std.time.ns_per_ms);
    }

    try std.testing.expect(ran_before_release);
}

test "deinit waits for in-flight join callers" {
    const pool = try ThreadPool.init(.{
        .allocator = std.testing.allocator,
        .thread_count = 2,
    });

    var a_started = std.atomic.Value(bool).init(false);
    var join_done = std.atomic.Value(bool).init(false);
    var result_a = std.atomic.Value(u64).init(0);
    var result_b = std.atomic.Value(u64).init(0);

    const caller = try std.Thread.spawn(.{}, struct {
        fn run(
            p: *ThreadPool,
            started: *std.atomic.Value(bool),
            done: *std.atomic.Value(bool),
            ra: *std.atomic.Value(u64),
            rb: *std.atomic.Value(u64),
        ) void {
            const result = p.join(
                struct {
                    fn a(flag: *std.atomic.Value(bool)) u64 {
                        flag.store(true, .release);
                        std.Thread.sleep(20 * std.time.ns_per_ms);
                        return 7;
                    }
                }.a,
                .{started},
                struct {
                    fn b() u64 {
                        return 11;
                    }
                }.b,
                .{},
            );
            ra.store(result.a, .release);
            rb.store(result.b, .release);
            done.store(true, .release);
        }
    }.run, .{ pool, &a_started, &join_done, &result_a, &result_b });

    while (!a_started.load(.acquire)) {
        std.atomic.spinLoopHint();
    }

    pool.deinit();
    caller.join();

    try std.testing.expect(join_done.load(.acquire));
    try std.testing.expectEqual(@as(u64, 7), result_a.load(.acquire));
    try std.testing.expectEqual(@as(u64, 11), result_b.load(.acquire));
}
