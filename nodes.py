import gc
import torch
import comfy.model_management as model_management

# Global state to track loop iterations across workflow executions
_loop_state = {}


def get_loop_state(loop_id):
    """Get the current state for a loop by ID."""
    return _loop_state.get(loop_id, None)


def set_loop_state(loop_id, state):
    """Set the state for a loop by ID."""
    _loop_state[loop_id] = state


def clear_loop_state(loop_id=None):
    """Clear loop state. If loop_id is None, clear all states."""
    global _loop_state
    if loop_id is None:
        _loop_state = {}
    elif loop_id in _loop_state:
        del _loop_state[loop_id]


class LoopStart:
    """
    Loop Start node - Initializes a loop with start and max iteration values.
    Outputs the current iteration index and a signal to control flow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_iterations": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
                "loop_signal": ("LOOP_SIGNAL",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "LOOP_SIGNAL")
    RETURN_NAMES = ("current_index", "start_index", "max_iterations", "loop_signal")
    FUNCTION = "execute"
    CATEGORY = "looping"

    def execute(self, start_index, max_iterations, loop_signal=None):
        if loop_signal is None:
            current_index = start_index
        else:
            current_index = loop_signal.get("current_index", start_index)

        signal = {
            "current_index": current_index,
            "start_index": start_index,
            "max_iterations": max_iterations,
            "is_active": current_index < (start_index + max_iterations)
        }

        return (current_index, start_index, max_iterations, signal)


class LoopEnd:
    """
    Loop End node - Handles iteration increment and memory cleanup.
    Provides memory-safe iteration with configurable cleanup options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_signal": ("LOOP_SIGNAL",),
            },
            "optional": {
                "passthrough": ("*",),
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "run_gc": ("BOOLEAN", {"default": True}),
                "unload_models": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LOOP_SIGNAL", "*", "BOOLEAN")
    RETURN_NAMES = ("loop_signal", "passthrough", "continue_loop")
    FUNCTION = "execute"
    CATEGORY = "looping"
    OUTPUT_NODE = True

    def execute(self, loop_signal, passthrough=None, clear_cuda_cache=True, run_gc=True, unload_models=False):
        current_index = loop_signal.get("current_index", 0)
        start_index = loop_signal.get("start_index", 0)
        max_iterations = loop_signal.get("max_iterations", 1)

        next_index = current_index + 1
        end_index = start_index + max_iterations
        continue_loop = next_index < end_index

        self._cleanup_memory(clear_cuda_cache, run_gc, unload_models)

        new_signal = {
            "current_index": next_index,
            "start_index": start_index,
            "max_iterations": max_iterations,
            "is_active": continue_loop
        }

        return (new_signal, passthrough, continue_loop)

    def _cleanup_memory(self, clear_cuda_cache, run_gc, unload_models):
        """Perform memory cleanup operations."""
        if unload_models:
            model_management.unload_all_models()
            model_management.soft_empty_cache()

        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if run_gc:
            gc.collect()


class LoopCondition:
    """
    Loop Condition node - Checks if the loop should continue.
    Useful for conditional branching based on loop state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_signal": ("LOOP_SIGNAL",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("should_continue", "current_index", "remaining")
    FUNCTION = "execute"
    CATEGORY = "looping"

    def execute(self, loop_signal):
        current_index = loop_signal.get("current_index", 0)
        start_index = loop_signal.get("start_index", 0)
        max_iterations = loop_signal.get("max_iterations", 1)

        end_index = start_index + max_iterations
        should_continue = current_index < end_index
        remaining = max(0, end_index - current_index - 1)

        return (should_continue, current_index, remaining)


class LoopIndex:
    """
    Loop Index node - Extracts iteration information from loop signal.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_signal": ("LOOP_SIGNAL",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("current_index", "iteration", "total_iterations", "remaining", "progress")
    FUNCTION = "execute"
    CATEGORY = "looping"

    def execute(self, loop_signal):
        current_index = loop_signal.get("current_index", 0)
        start_index = loop_signal.get("start_index", 0)
        max_iterations = loop_signal.get("max_iterations", 1)

        iteration = current_index - start_index
        remaining = max(0, max_iterations - iteration - 1)
        progress = (iteration + 1) / max_iterations if max_iterations > 0 else 1.0

        return (current_index, iteration, max_iterations, remaining, progress)


class MemoryCleanup:
    """
    Standalone Memory Cleanup node - Can be used anywhere in the workflow.
    Provides aggressive memory optimization options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),
            },
            "optional": {
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "run_gc": ("BOOLEAN", {"default": True}),
                "unload_models": ("BOOLEAN", {"default": False}),
                "aggressive_cleanup": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "looping/utils"

    def execute(self, trigger, clear_cuda_cache=True, run_gc=True, unload_models=False, aggressive_cleanup=False):
        if unload_models:
            model_management.unload_all_models()
            model_management.soft_empty_cache()

        if aggressive_cleanup:
            model_management.cleanup_models()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if run_gc:
            gc.collect()
            if aggressive_cleanup:
                gc.collect()
                gc.collect()

        return (trigger,)


class IntIterator:
    """
    Simple Integer Iterator - Generates integers from start to end.
    Useful for batch processing without full loop control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "end": ("INT", {"default": 10, "min": -10000, "max": 10000, "step": 1}),
                "step": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "current_iteration": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("value", "iteration", "has_more")
    FUNCTION = "execute"
    CATEGORY = "looping"

    def execute(self, start, end, step, current_iteration):
        value = start + (current_iteration * step)
        has_more = (value + step) <= end if step > 0 else (value + step) >= end

        return (value, current_iteration, has_more)


class AutoLoopStart:
    """
    Auto Loop Start - Automatically tracks iteration state using a unique loop ID.
    Works with Auto Queue to create true iterative loops.

    Usage:
    1. Connect this node at the start of your workflow
    2. Enable "Auto Queue" in ComfyUI
    3. The loop will automatically run from start_index to max_iterations
    4. Reset the loop by changing the loop_id or using the reset input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_id": ("STRING", {"default": "loop_1", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "BOOLEAN", "LOOP_SIGNAL")
    RETURN_NAMES = ("current_index", "iteration", "remaining", "progress", "is_last", "loop_signal")
    FUNCTION = "execute"
    CATEGORY = "looping"

    @classmethod
    def IS_CHANGED(cls, loop_id, start_index, max_iterations, reset=False):
        state = get_loop_state(loop_id)
        if state is None or reset:
            return float("nan")
        return state.get("current_index", 0)

    def execute(self, loop_id, start_index, max_iterations, reset=False):
        state = get_loop_state(loop_id)

        if state is None or reset:
            current_index = start_index
            state = {
                "current_index": current_index,
                "start_index": start_index,
                "max_iterations": max_iterations,
            }
            set_loop_state(loop_id, state)
        else:
            current_index = state.get("current_index", start_index)

        iteration = current_index - start_index
        end_index = start_index + max_iterations
        remaining = max(0, end_index - current_index - 1)
        progress = (iteration + 1) / max_iterations if max_iterations > 0 else 1.0
        is_last = current_index >= (end_index - 1)

        signal = {
            "loop_id": loop_id,
            "current_index": current_index,
            "start_index": start_index,
            "max_iterations": max_iterations,
            "is_active": current_index < end_index
        }

        return (current_index, iteration, remaining, progress, is_last, signal)


class AutoLoopEnd:
    """
    Auto Loop End - Increments the loop counter and performs memory cleanup.
    Connect any output that you want to ensure completes before the loop advances.

    Memory cleanup options:
    - clear_cuda_cache: Free unused CUDA memory
    - run_gc: Run Python garbage collection
    - unload_models: Unload all models from VRAM (use sparingly)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_signal": ("LOOP_SIGNAL",),
            },
            "optional": {
                "passthrough": ("*",),
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "run_gc": ("BOOLEAN", {"default": True}),
                "unload_models": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN", "INT")
    RETURN_NAMES = ("passthrough", "should_continue", "next_index")
    FUNCTION = "execute"
    CATEGORY = "looping"
    OUTPUT_NODE = True

    def execute(self, loop_signal, passthrough=None, clear_cuda_cache=True, run_gc=True, unload_models=False):
        loop_id = loop_signal.get("loop_id", "default")
        current_index = loop_signal.get("current_index", 0)
        start_index = loop_signal.get("start_index", 0)
        max_iterations = loop_signal.get("max_iterations", 1)

        next_index = current_index + 1
        end_index = start_index + max_iterations
        should_continue = next_index < end_index

        self._cleanup_memory(clear_cuda_cache, run_gc, unload_models)

        state = {
            "current_index": next_index,
            "start_index": start_index,
            "max_iterations": max_iterations,
        }
        set_loop_state(loop_id, state)

        if not should_continue:
            clear_loop_state(loop_id)

        return (passthrough, should_continue, next_index)

    def _cleanup_memory(self, clear_cuda_cache, run_gc, unload_models):
        """Perform memory cleanup operations."""
        if unload_models:
            model_management.unload_all_models()
            model_management.soft_empty_cache()

        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if run_gc:
            gc.collect()


class LoopBreak:
    """
    Loop Break - Stops the loop early based on a condition.
    When break_condition is True, the loop state is cleared.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_signal": ("LOOP_SIGNAL",),
                "break_condition": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "passthrough": ("*",),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("passthrough", "did_break")
    FUNCTION = "execute"
    CATEGORY = "looping"
    OUTPUT_NODE = True

    def execute(self, loop_signal, break_condition, passthrough=None):
        if break_condition:
            loop_id = loop_signal.get("loop_id", "default")
            clear_loop_state(loop_id)

        return (passthrough, break_condition)


class LoopReset:
    """
    Loop Reset - Resets a specific loop to its initial state.
    Useful for restarting loops or clearing stuck state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_id": ("STRING", {"default": "loop_1", "multiline": False}),
                "trigger": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "looping/utils"
    OUTPUT_NODE = True

    def execute(self, loop_id, trigger):
        clear_loop_state(loop_id)
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "LoopStart": LoopStart,
    "LoopEnd": LoopEnd,
    "LoopCondition": LoopCondition,
    "LoopIndex": LoopIndex,
    "MemoryCleanup": MemoryCleanup,
    "IntIterator": IntIterator,
    "AutoLoopStart": AutoLoopStart,
    "AutoLoopEnd": AutoLoopEnd,
    "LoopBreak": LoopBreak,
    "LoopReset": LoopReset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopStart": "Loop Start (Memory Safe)",
    "LoopEnd": "Loop End (Memory Safe)",
    "LoopCondition": "Loop Condition",
    "LoopIndex": "Loop Index",
    "MemoryCleanup": "Memory Cleanup",
    "IntIterator": "Integer Iterator",
    "AutoLoopStart": "Auto Loop Start",
    "AutoLoopEnd": "Auto Loop End",
    "LoopBreak": "Loop Break",
    "LoopReset": "Loop Reset",
}
