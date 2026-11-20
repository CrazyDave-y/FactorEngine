# FactorEngine 许可说明

## 默认许可证与适用范围

自引入本文件的许可证变更提交起，FactorEngine 当前及未来由仓库所有者 [CrazyDave-y](https://github.com/CrazyDave-y) 拥有版权或已取得相应重新授权权利、且未另行注明的源码、文档和配置，默认采用 [PolyForm Strict License 1.0.0](LICENSE)。未来版本如另有明确许可声明，以该声明为准。打包元数据使用自定义 SPDX 表达式标识 `LicenseRef-PolyForm-Strict-1.0.0`，指向随包提供的许可证全文；不将其误写为 SPDX 标准列表中已收录的标识。

`LICENSE` 保留 [PolyForm 官方标准文本](https://polyformproject.org/licenses/strict/1.0.0.txt)，本说明不增删其条款。PolyForm Strict 属于源码可见许可证，不应描述为允许自由修改和再分发的开源许可证。

PolyForm Strict 允许其规定的非商业用途，包括符合条件的个人研究、实验、测试及非商业机构使用。其非商业机构条款有特定范围，应阅读原文，不应将本说明理解为全面禁止任何有资金支持的研究活动。

PolyForm Strict 本身不授予分发软件、修改软件或基于软件制作新作品的版权许可，即使相关行为出于非商业目的。超出其允许用途的商业使用，以及需要上述额外权利的行为，须另行取得相应权利人的书面许可。法律赋予的合理使用等权利、其他有效许可和历史授权不受本说明限制。

商业授权请联系 [CrazyDave-y](https://github.com/CrazyDave-y)。任何另行商业许可的范围、费用及条件，均以实际签署的协议为准；本文件本身不是商业许可。

## 历史 Apache-2.0 授权不被追溯撤销

本次变更前的 `main` 分支提交为：

- 最近历史提交：[`6d7c3a0cae0c04bad9125718b684c608b950a2f3`](https://github.com/CrazyDave-y/FactorEngine/commit/6d7c3a0cae0c04bad9125718b684c608b950a2f3)。
- 初始公开源码提交：[`3a7050866fd2117c2f94c581ded18f8927573099`](https://github.com/CrazyDave-y/FactorEngine/commit/3a7050866fd2117c2f94c581ded18f8927573099)。
- 该历史版本的 [Apache-2.0 许可证全文](https://github.com/CrazyDave-y/FactorEngine/blob/6d7c3a0cae0c04bad9125718b684c608b950a2f3/LICENSE) 保留在 Git 历史中。

已经按 Apache-2.0 合法发布的版本及代码，其既有许可继续按 Apache-2.0 的条款和条件有效。本次变更不追溯撤销这些权利，也不要求合法持有者放弃原许可。持有者仍可在遵守原许可的前提下使用、修改、商业使用和分发相应历史代码。

仅替换 `LICENSE`、复制相同代码或提高版本号，不会消除已经授出的 Apache-2.0 权利。因此，当前版本中与历史版本相同的代码，并不因本次变更就只能按 PolyForm Strict 使用；本次默认许可声明不能限制依原许可合法取得和使用的代码。对于后续新增内容，只有在权利人有权授权且未另行授出其他许可的范围内，才适用相应的新许可限制。

本次操作保留 Git 历史，不删除旧版本、重写提交或尝试撤销已有 fork 的合法授权。Apache-2.0 的版权、专利、再分发和归属声明要求仍适用于依该许可行使权利的行为，具体见 [Apache 官方文本](https://www.apache.org/licenses/LICENSE-2.0)。

## 第三方代码、依赖和贡献

本默认许可不重新授权第三方依赖、第三方代码或没有取得重新授权权利的外部贡献。这些内容继续适用其原许可证、版权及归属声明；不得以本仓库根目录的许可证替代或删除它们。仅拥有仓库管理权限，或仅接受过 Apache-2.0 贡献，并不能据此断言拥有贡献者的版权或排他的重新授权权利。

在本次变更前的提交 `6d7c3a0cae0c04bad9125718b684c608b950a2f3` 上，检查了全部 105 个受版本控制文件、两次提交的作者记录及仓库 PR 记录。两次提交均关联到 `CrazyDave-y`，未发现合入的外部 PR、第三方版权头、独立 NOTICE/COPYING、内嵌第三方源码目录或子模块。此检查只反映仓库中可见的记录，不能单独证明完整的版权归属或排除未披露的第三方来源。

当前代码通过导入、头文件或链接使用以下外部组件；它们没有因本次变更而改用 PolyForm Strict：

| 组件 | 当前使用位置 | 许可边界 |
| --- | --- | --- |
| NumPy | Python 运行时导入，`pyproject.toml` 中的依赖 | 适用所用 NumPy 版本自身的许可证和声明 |
| NVIDIA CUDA Toolkit / Runtime | CMake 的 `CUDA::cudart`，CUDA 头文件 | 适用所用 Toolkit / Runtime 的相应条款 |
| CUB / CCCL | `cuda/temporal.cuh`、`cuda/resident_runtime.cuh`、`cuda/reference/cub_pair.cu` 中的外部头文件 | 适用所用 CUB / CCCL 版本自身的许可证和声明 |
| C++ 标准库及 OpenMP 实现 | C++ 标准库头文件、`omp.h`、`OpenMP::OpenMP_CXX` | 适用所用工具链及运行库的相应条款 |
| 构建和开发工具 | setuptools、pytest、Pyright、Ruff 等 | 适用各工具自身的许可证 |

以后引入外部代码或贡献时，必须保留原有声明并明确适用许可。若未取得重新授权所需权利，应在相应文件或目录中明确例外，不得将其直接宣称为受本默认许可排他控制。

## Public 仓库与平台权限

仓库保持公开。GitHub 平台条款另行授予平台及用户相应权利，包括查看和在平台内 fork；本说明不撤销这些权利。平台内 fork 不等于获得超出适用许可的任意修改、再分发或商业使用授权。具体平台权利见 [GitHub Terms of Service，D 节](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#d-user-generated-content)。

本文件是许可范围及变更记录说明；具体行为应依据 `LICENSE`、适用的历史或第三方许可证、平台条款和另行协议判断。
