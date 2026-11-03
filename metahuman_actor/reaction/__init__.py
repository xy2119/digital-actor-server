"""Empathetic-listener reaction retrieval over the Express4D clip library.

Player input -> LLM bridge (English situation + listener function + intensity)
-> tag-routed vector retrieval (bge-small MaxSim) -> metadata rerank ->
temperature sampling -> ARKit blendshape clip streamed to the game.
"""
