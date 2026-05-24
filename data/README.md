# Data

Experiment inputs bundled with the repository. Large external datasets (COCO, OpenPSG, GQA) are **not** included — download them separately if you use `--coco`, `--psg`, or `--gqa`.

## Layout

```
data/
├── prompts/          # JSON prompt lists for training and evaluation
│   ├── prompts_paper.json      # 4 paper Sec 5.2 prompts (training)
│   ├── prompts_holdout.json    # 4 unseen prompts (generalization eval)
│   └── prompts_single.json     # 1-prompt diagnostic ("A green colored rabbit.")
└── README.md         # this file
```

## Prompt files

| File | Contents | Used in |
|------|----------|---------|
| `prompts/prompts_paper.json` | Rabbit, cat+dog, four wolves, dog on moon | `submit_imagereward.sh`, paper reproduction |
| `prompts/prompts_holdout.json` | Elephant, horse+sheep, five birds, penguin on beach | Holdout eval in `submit_imagereward.sh` |
| `prompts/prompts_single.json` | `"A green colored rabbit."` | `submit_imagereward_single.sh`, quick smoke tests |

Each file is a JSON **list of strings**, e.g.:

```json
[
  "A green colored rabbit.",
  "A cat and a dog."
]
```

## Eval results (not raw training data)

Pre-computed evaluation tables and plots from HPC runs live under:

- `eval-single-prompt/` — job 484798 (single-prompt train + eval)
- `eval-four-prompts/` — job 484796 (4-prompt train + holdout eval)

These folders contain CSV/JSON/PNG reports, not model checkpoints.

## Reference images (`src/data/`)

Small demo images used by the hierarchical CLIP reward code in `src/` (`ref1.png`, `ref2.png`).
