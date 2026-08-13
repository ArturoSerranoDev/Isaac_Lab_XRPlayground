# Shared assets

Cross-project assets shared between the Isaac (`Isaac_XRPlayground/`) and Unity
(`Unity_XRPlayground/`, added later) sides of this repository.

Suggested layout:

```text
assets/
  usd/        # USD scenes/props authored in Isaac Sim (File -> Save As here)
  meshes/     # raw meshes (.obj, .fbx, .stl) before conversion
  textures/   # shared textures / materials
```

## Notes

- These files are tracked by the umbrella git repo at `Isaac_Lab_XRPlayground/`.
- Isaac Lab does **not** auto-discover assets here. Reference them explicitly in
  a task config, e.g. `UsdFileCfg(usd_path=".../assets/usd/my_scene.usd")`.
- Keep large binaries reasonable; consider Git LFS if they grow.
