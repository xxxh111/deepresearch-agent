"""pytest 公共配置。

把项目根加进 sys.path，这样测试里可以直接 `from graph import ...`，
不需要把项目 pip install 成包。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
