# Local Data Directory

Everything below `data/` is local-only except this document and deliberately
small files under `data/samples/`.

Recommended layout:

```text
data/
├── raw/
│   └── <dataset>/<SYMBOL>/20s/YYYY-MM-DD.csv
├── runtime/
│   └── bot.sqlite3
├── manifests/
│   └── generated/
└── samples/
```

Do not commit full market datasets, exchange exports, account data, SQLite
databases, or generated manifests. Use the scripts under `scripts/data/` to
build local datasets. If a reproducible public sample is needed, minimize and
review it before placing it in `data/samples/`.
