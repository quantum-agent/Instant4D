# Vendored third-party code

This fork intentionally vendors formerly nested submodule dependencies directly into the tree so the parent repo only tracks a single submodule boundary (third_party/instant4d).

Vendored sources captured during the 2026-05-13 flattening pass:

- SLAM/mega-sam/ ← https://github.com/mega-sam/mega-sam @ 2291f0155f93ded172db195d33c8639896f0b3cc
- SLAM/mega-sam/base/ ← https://github.com/mega-sam/base @ a26d54b7d85f0cde50c448778d09ba6952790c65
- SLAM/mega-sam/base/thirdparty/lietorch/ ← https://github.com/princeton-vl/lietorch @ 2d2c3347314606f080ccfb5dbab758bf4e22c565
- SLAM/mega-sam/base/thirdparty/eigen/ ← https://gitlab.com/libeigen/eigen.git @ 3d4ba855e014987cad86d62a8dff533492255695
- submodule/fussed-ssim/ ← https://github.com/rahul-goel/fused-ssim @ b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3

Local compatibility fixes already applied in this fork are preserved in the vendored copies.
