from . import compat  # noqa: F401 — monkey-patch nuscenes/pandas2 兼容性,必须最先 import
from .datasets import *
from .models import *
from .apis import *
from .core.evaluation import *
from .core.hook import *  # noqa: F401,F403 — 注册 NaNStopHook 等自定义 hook
