Vendored subset of [GenZ-ICP](https://github.com/cocel-postech/genz-icp) (MIT) at commit 674ccce:
`core/VoxelHashMap.{hpp,cpp}` and `core/Registration.{hpp,cpp}`, unmodified.

Header-only dependencies vendored next to it so the build needs no network or system packages:
- `sophus/` — Sophus (MIT), the pre-fmt header-only revision also used by hikari_loclite / lightning-lm.
- `tsl/` — tsl::robin_map (MIT), upstream Tessil/robin-map as fetched by the GenZ-ICP build.
