from __future__ import annotations

"""从训练配置文件中加载检测白名单。

支持两类常见输入：
1. 直接传数据集 YAML（根节点包含 `names`）；
2. 传训练配置 YAML（根节点 `data` 指向数据集 YAML，或 `data` 内嵌 `names`）。

该模块的目标是把“推理白名单”和“训练类别定义”统一到同一个来源，
减少手工维护两份类别列表导致的偏差。
"""

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover - 仅在极端环境缺少依赖时触发
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


def load_whitelist_from_config(
    config_path: Path,
    field_path: str = "names",
) -> tuple[str, ...]:
    """从配置文件加载类别白名单。

    参数:
        config_path: 配置文件路径（支持相对/绝对路径）。
        field_path: 类别字段路径，默认 `names`。支持点分路径，例如 `data.names`。

    返回:
        tuple[str, ...]: 去重后的类别名元组（保留原始大小写）。

    异常:
        RuntimeError: 缺少 YAML 解析依赖时抛出。
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 文件可读但无法解析出有效类别列表时抛出。
    """
    config_path = Path(config_path).expanduser().resolve()
    root = _load_yaml_file(config_path)

    names = _extract_names(root, field_path)
    if names:
        return _dedupe_names(names)

    # 训练配置常见写法：data: xxx.yaml
    if isinstance(root, dict):
        data_node = root.get("data")
        if isinstance(data_node, str) and data_node.strip():
            data_yaml_path = _resolve_data_yaml_path(config_path, data_node)
            data_doc = _load_yaml_file(data_yaml_path)
            names = _extract_names(data_doc, field_path)
            if names:
                return _dedupe_names(names)
            # 若 field_path 未命中，兜底尝试根 `names`。
            names = _extract_names(data_doc, "names")
            if names:
                return _dedupe_names(names)
        elif isinstance(data_node, dict):
            names = _extract_names(data_node, field_path)
            if names:
                return _dedupe_names(names)
            names = _extract_names(data_node, "names")
            if names:
                return _dedupe_names(names)

    raise ValueError(
        "未在配置中解析到类别白名单。请检查配置是否包含 `names`，"
        "或训练配置中的 `data` 是否正确指向数据集 YAML。"
    )


def _load_yaml_file(path: Path) -> Any:
    """读取并解析 YAML 文件。"""
    if yaml is None:
        raise RuntimeError(
            "当前环境缺少 YAML 解析依赖（PyYAML）。"
            f" 原始错误: {_YAML_IMPORT_ERROR}"
        )
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_data_yaml_path(config_path: Path, data_value: str) -> Path:
    """解析训练配置中的 data 路径。

    - 若 data_value 是绝对路径，直接使用；
    - 若是相对路径，优先按训练配置文件所在目录解析；
    - 若不存在，再尝试按当前工作目录解析。
    """
    raw = Path(data_value).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    by_config_dir = (config_path.parent / raw).resolve()
    if by_config_dir.exists():
        return by_config_dir
    return raw.resolve()


def _extract_names(root: Any, field_path: str) -> list[str]:
    """按字段路径提取并归一化 names。"""
    node = _select_by_path(root, field_path)
    names = _normalize_names(node)
    if names:
        return names
    if field_path != "names":
        return _normalize_names(_select_by_path(root, "names"))
    return []


def _select_by_path(root: Any, path: str) -> Any:
    """通过点分路径读取节点，例如 `data.names`。"""
    if not path:
        return root
    node = root
    for token in (part.strip() for part in path.split(".")):
        if not token:
            continue
        if isinstance(node, dict):
            if token in node:
                node = node[token]
                continue
            # 兼容键为数字字符串的场景
            if token.isdigit() and int(token) in node:
                node = node[int(token)]
                continue
        return None
    return node


def _normalize_names(node: Any) -> list[str]:
    """将 names 节点转换为字符串列表。"""
    if node is None:
        return []
    if isinstance(node, (list, tuple)):
        return [str(item).strip() for item in node if str(item).strip()]
    if isinstance(node, dict):
        ordered: list[tuple[tuple[int, int], str]] = []
        for index, (key, value) in enumerate(node.items()):
            name = str(value).strip()
            if not name:
                continue
            order_key = _dict_key_order(key=key, fallback_index=index)
            ordered.append((order_key, name))
        ordered.sort(key=lambda item: item[0])
        return [item[1] for item in ordered]
    if isinstance(node, str):
        name = node.strip()
        return [name] if name else []
    return []


def _dict_key_order(key: Any, fallback_index: int) -> tuple[int, int]:
    """给字典键生成稳定排序键。

    `names` 若是 `{0: "cup", 1: "bottle"}`，按数值顺序排序；
    若键不可转数字，则保持原出现顺序。
    """
    try:
        return (0, int(key))
    except Exception:
        return (1, int(fallback_index))


def _dedupe_names(names: list[str]) -> tuple[str, ...]:
    """按大小写不敏感方式去重并保持首次出现顺序。"""
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(name)
    return tuple(output)
