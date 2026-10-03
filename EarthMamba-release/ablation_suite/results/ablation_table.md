# EarthMamba Leave-One-Out 消融结果

> 设计：EarthMamba-base + 3×w/o + Full（Full 引用主表）。
> 由 `scripts/collect_results.py` 自动生成。

| Variant | ARMG | SparseSSM | Latent Graph | DFC15 macro-mAP | INRIA mIoU | DIOR mAP@0.5 |
|---------|:----:|:---------:|:------------:|----------------:|-----------:|-------------:|
| EarthMamba-base |  |  |  |  |  |  |
| w/o ARMG |  | ✓ | ✓ |  |  |  |
| w/o SparseSSM | ✓ |  | ✓ |  |  |  |
| w/o Latent Graph | ✓ | ✓ |  |  |  |  |
| EarthMamba full | ✓ | ✓ | ✓ | 97.54 | 85.59 | 66.27 |

## 论文表述建议

- 不写 *strong synergistic effects*；写 *each component contributes and their combination achieves the best overall transferability*。
- Baseline 为 **EarthMamba-base**（同 stage/channel/pretrain/下游协议，仅关闭三个模块），不是 VMamba。
- SparseSSM 列：主 checkpoint 用 `SS2D_Mamba3` 时 `WO_SPARSE` 的 SSM 路径与 FULL 相同，见 `CODE_REALITY.md` / 附录。

## 明细

### BASE

### WO_ARMG

### WO_SPARSE

### WO_GRAPH

### FULL
- dfc15_map: 97.54
- dior_map50: 66.27
- inria_miou: 85.59
- source: main_table_reference
