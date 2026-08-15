# Agibot A2D USD for Unity

Copy from Isaac Nucleus (Isaac Sim Content Browser or Nucleus mount):

```
IsaacLab/Nucleus/Robots/Agibot/A2D/A2D_physics.usd
```

Into this repo:

```
Unity_XRPlayground/Assets/_Project/Features/Robots/AgibotA2D/USD/A2D_physics.usd
```

Also mirror here for reference:

```
assets/usd/agibot/A2D/A2D_physics.usd
```

After import in Unity, run **Rebuild Link Map** on `AgibotLinkMap` and verify body names against `names_agibot.py`. Adjust `AgibotLinkMap.expectedLinks` if the USD uses different link names.

Wall-mounted pose in Unity matches Isaac: robot root at Isaac `(0, -0.78, 0)` → Unity via `XrFrameConverter.IsaacPosToUnity`.
