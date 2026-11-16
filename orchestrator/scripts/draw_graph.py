"""Write the graph as Mermaid text (docs/graph.mmd) and, if online, as a PNG (docs/graph.png).

GitHub renders the Mermaid text inside Markdown, so docs/graph.mmd can be pasted
into a README in a ```mermaid block.
"""
from pathlib import Path

from navigator.graph import build_graph

out = Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)
g = build_graph().get_graph()
(out / "graph.mmd").write_text(g.draw_mermaid(), encoding="utf-8")
print("wrote", out / "graph.mmd")
try:
    (out / "graph.png").write_bytes(g.draw_mermaid_png())   # renders via the mermaid.ink web service
    print("wrote", out / "graph.png")
except Exception as e:  # offline, or the service is unreachable
    print("PNG skipped:", type(e).__name__, e)
