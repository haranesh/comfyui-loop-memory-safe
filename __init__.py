"""
ComfyUI Loop Memory Safe
========================

A ComfyUI custom node pack providing memory-safe loop functionality.

Features:
- Loop Start/End nodes for iterating workflows multiple times
- Auto Loop nodes that work with ComfyUI's Auto Queue feature
- Configurable memory cleanup after each iteration
- CUDA cache clearing and garbage collection
- Optional model unloading between iterations
- Progress tracking and loop condition nodes
- Loop break and reset functionality

Nodes included:
- Loop Start (Memory Safe): Initialize loop with start index and max iterations
- Loop End (Memory Safe): Handle iteration and memory cleanup
- Auto Loop Start: Automatically tracks state for use with Auto Queue
- Auto Loop End: Increments counter and cleans memory for Auto Queue loops
- Loop Condition: Check if loop should continue
- Loop Index: Extract iteration information
- Loop Break: Stop loop early based on condition
- Loop Reset: Reset a loop to initial state
- Memory Cleanup: Standalone memory optimization node
- Integer Iterator: Simple integer sequence generator

Usage with Auto Queue:
1. Add "Auto Loop Start" node and set loop_id, start_index, and max_iterations
2. Connect your workflow
3. Add "Auto Loop End" node and connect the loop_signal
4. Enable "Auto Queue" in ComfyUI's queue settings
5. Queue the workflow - it will automatically iterate until complete
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "1.0.0"
