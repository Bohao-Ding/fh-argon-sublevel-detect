# Phase 3 v1 运行收据

## 命令

```powershell
python run_phase3.py --workers auto --surrogate-replicates 2000 --injection-replicates 500
```

## 状态

- 退出状态：0
- 工作进程：8
- 总墙钟时间：18.017 s
- 条件代理任务时间和/墙钟比：7.103
- 注入任务时间和/墙钟比：4.433
- 输入数据 SHA-256：`FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3`
- 哈希清单：`results/phase3_v1/artifacts_sha256.csv`
- 哈希清单自身 SHA-256：`EF542785001C2C9824C82AD96CBA962F99C73BBEC9F7D818CD0DA0C530F4F801`

## 验证

- 测试：`13 passed in 2.84s`
- 哈希复核：清单 29 项，`0` 项不一致
- 条件代理样本：`32,000` 行；每任务 `2,000` 次
- 注入选择样本：`24,000` 行；每任务 `500` 次
- 六张关键数值表：所有数值字段均为有限值
- 旧稿 SHA-256：`3C7343622B4E3FCF2FCB0223E8E21976153D56C288F89D9075C3DF3E1B201145`，与 Phase 3 前一致

## 证据身份

内部开发性稳健性分析；不是独立实验确认、物理通道识别或仪器分辨率标定。
