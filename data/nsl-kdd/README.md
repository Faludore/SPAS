# NSL-KDD dataset

The dataset files are intentionally not included in this repository.

Place the original NSL-KDD text files in this directory using the following exact names:

```text
KDDTrain+.txt
KDDTest+.txt
```

Expected repository layout:

```text
data/
└── nsl-kdd/
    ├── KDDTrain+.txt
    ├── KDDTest+.txt
    └── README.md
```

Each record is expected to contain the 41 NSL-KDD input features followed by:

1. the textual class label;
2. the difficulty-level field.

The experiment scripts convert the textual labels into a binary target:

- `normal` → `0`;
- every attack label → `1`.

Do not rename the files unless the paths in all three scripts are updated accordingly.
