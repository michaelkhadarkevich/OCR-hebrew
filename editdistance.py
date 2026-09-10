from __future__ import annotations

from collections.abc import Sequence


def eval(source: Sequence[object], target: Sequence[object]) -> int:
    if source == target:
        return 0
    if len(source) < len(target):
        source, target = target, source
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
