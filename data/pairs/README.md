# Image pairs

Put `(source, target)` image pairs in this directory. Both images should be
square (we resize to 512 x 512 by default) and ideally roughly
layout-aligned -- the paper's setting uses depth-aligned faces and
coarsely corresponded scenes.

Suggested layout:

```
data/pairs/
├── face_pairs/
│   ├── 0001/
│   │   ├── source.png
│   │   └── target.png
│   └── 0002/
│       ├── source.png
│       └── target.png
├── morph4data/              # Morph4Data benchmark pairs (download from FreeMorph)
└── morphbench/              # MorphBench pairs (download from IMPUS / DiffMorpher)
```

The default shell scripts read from `./data/pairs/example_source.png` and
`./data/pairs/example_target.png`; override via the `SOURCE_IMAGE` and
`TARGET_IMAGE` environment variables.
