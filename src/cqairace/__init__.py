"""CQ AI Race 参赛智能体包。

分层约定：官方基础设施（gRPC 协议、生命周期、TongSim 客户端）来自
arenaagentpro（official_source/gitee/baseline-agent，editable 挂载，不修改）；
本包只实现自研"大脑层"：agent 主类、按任务模型路由、任务专属 prompt。
设计文档见 docs/solution/architecture.md。
"""

__version__ = "0.1.0"
