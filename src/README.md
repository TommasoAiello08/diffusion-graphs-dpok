# Research code (`src/`)

Earlier prototypes for **hierarchical scene-graph CLIP rewards** and **FK steering** with ImageReward. Not imported by the production training script `dpok_imagereward.py`.

## Layout

```
src/
├── simple_hier_clip_reward.py   # Hierarchical CLIP reward demo
├── message_passing_clip_reward.py
├── refgraphs.py / refgraphs2.py # Synthetic scene-graph generators
├── example.py                   # Reward-guided sampling demo
├── example_t2i.py               # Plain SD1.5 text-to-image
├── example_fksteering.py        # FK steering with diffusers pipeline
├── rewards/                     # CLIP scorer utilities
├── fk_steering/                 # FK-Steering diffusers integration
├── viz/                         # Graph visualization helpers
└── data/                        # Small reference images (ref1.png, ref2.png)
```

## Run examples

```bash
cd src
python simple_hier_clip_reward.py
python example_t2i.py
```

Requires dependencies from the root `requirements.txt`.
