"""Name -> class registries so that every component can be selected from a config file."""
from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        # 输入：kind；输出：None；初始化实例及校验参数；继承：Registry(object)。
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        # 输入：name；输出：将组件类加入注册表的装饰器；继承：Registry(object)。
        def deco(cls):
            # 输入：待注册组件类 cls（闭包读取 name）；输出：注册后的原组件类；继承：无（register 内部函数）。
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' registered twice")
            cls.registry_name = name
            self._items[name] = cls
            return cls

        return deco

    def build(self, name: str, params: dict | None = None):
        # 输入：name、params；输出：按名称和参数构造的组件；继承：Registry(object)。
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](**(params or {}))

    def names(self) -> list[str]:
        # 输入：实例状态；输出：已注册名称列表；继承：Registry(object)。
        return sorted(self._items)

    def describe(self) -> dict[str, str]:
        # 输入：实例状态；输出：注册组件说明字典；继承：Registry(object)。
        return {n: (c.__doc__ or "").strip().splitlines()[0] if c.__doc__ else "" for n, c in sorted(self._items.items())}


UNCERTAINTY_MODELS = Registry("uncertainty model")
MATCHERS = Registry("matcher")
PROPAGATIONS = Registry("uncertainty propagation")
OPTIMIZERS = Registry("global optimizer")
