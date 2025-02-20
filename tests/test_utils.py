import os
import pytest
from src.utils import setup_logging, set_random_seed

def test_setup_logging(tmp_path):
    """测试日志设置功能"""
    logger = setup_logging(tmp_path)
    assert logger is not None
    assert os.path.exists(os.path.join(tmp_path, "train.log"))

def test_set_random_seed():
    """测试随机种子设置功能"""
    # 设置种子不应该引发异常
    try:
        set_random_seed(42)
        assert True
    except Exception as e:
        pytest.fail(f"设置随机种子时发生异常: {str(e)}")

# 添加更多测试用例... 