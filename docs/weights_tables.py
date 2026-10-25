"""Sphinx directive rendering pretrained-weights tables straight from ``Weights.meta``.

``.. weights-table:: detection`` lists every registered weights entry for
that task, so the published numbers can never drift from the code. Like
torchvision's model tables, the benchmark columns come from ``_metrics``
and each entry's ``_docs`` / ``source`` / ``license`` follow the table.
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from torchocr.models import get_model_weights, list_models

COLUMNS = {
    "detection": [("ICDAR2015-test", "hmean", "ICDAR-2015 hmean")],
    "recognition": [
        ("ICDAR2015-test-gt-quads", "word_accuracy", "ICDAR-2015 word acc."),
        ("TotalText-test-gt-quads", "word_accuracy", "Total-Text word acc."),
    ],
}


def weights_for(task: str) -> list:
    seen, entries = set(), []
    for name in list_models():
        enum = get_model_weights(name)
        if enum is None or enum in seen:
            continue
        seen.add(enum)
        entries += [w for w in enum if w.meta.get("task") == task]
    return entries


def render(task: str) -> list[str]:
    columns = COLUMNS[task]
    header = ["Weights", "Params"] + [label for _, _, label in columns]
    rows = []
    for w in weights_for(task):
        metrics = w.meta.get("_metrics", {})
        values = [f"{metrics[ds][key]:.3f}" if key in metrics.get(ds, {}) else "—" for ds, key, _ in columns]
        rows.append([f"``{type(w).__name__}.{w.name}``", f"{w.meta['num_params'] / 1e6:.1f} M"] + values)
    lines = [".. list-table::", "   :header-rows: 1", ""]
    for row in [header] + rows:
        lines.append(f"   * - {row[0]}")
        lines += [f"     - {cell}" for cell in row[1:]]
    lines.append("")
    for w in weights_for(task):
        docs = w.meta.get("_docs", "").strip().replace("**", r"\*\*")  # code such as f(**kwargs), not bold
        lines += [f"``{type(w).__name__}.{w.name}``", "   " + docs, ""]
        lines += [f"   *Source:* {w.meta['source']}", "", f"   *License:* {w.meta['license']}", ""]
    return lines


class WeightsTable(Directive):
    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        container = nodes.container()
        self.state.nested_parse(StringList(render(self.arguments[0])), self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive("weights-table", WeightsTable)
    return {"parallel_read_safe": True}
