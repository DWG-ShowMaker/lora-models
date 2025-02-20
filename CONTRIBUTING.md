# Contributing to Qwen-LoRA-Project

我们非常欢迎您为Qwen-LoRA-Project做出贡献！

## 如何贡献

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

## 代码风格

- 遵循PEP 8 Python代码风格指南
- 使用类型注解
- 为函数和类添加文档字符串
- 保持代码简洁明了

## 提交Pull Request前的检查清单

- [ ] 代码符合项目的代码风格
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 所有测试都通过了
- [ ] 更新了requirements.txt（如果添加了新的依赖）

## Bug报告

请使用Issue模板提交bug报告，并包含以下信息：

- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（操作系统、Python版本等）
- 相关日志或错误信息

## 功能请求

如果您有新功能的想法，请：

1. 首先检查是否已经有相关的Issue或PR
2. 创建一个新的Issue描述您的想法
3. 讨论可行性和实现方案
4. 实现并提交PR

## 开发流程

1. 克隆仓库
```bash
git clone https://github.com/DWG-ShowMaker/lora-models.git
cd lora-models
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行测试
```bash
python -m pytest
```

## 分支策略

- `main`: 稳定版本分支
- `develop`: 开发分支
- `feature/*`: 新功能分支
- `bugfix/*`: 错误修复分支
- `release/*`: 发布准备分支

## 版本发布流程

1. 更新版本号
2. 更新CHANGELOG.md
3. 更新文档
4. 创建发布标签
5. 发布到PyPI（如果适用）

## 许可证

通过提交代码，您同意您的贡献将采用MIT许可证。

## 联系方式

如有任何问题，请通过Issue或讨论区联系我们。

感谢您的贡献！ 