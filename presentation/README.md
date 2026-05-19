# G1 Perceptive Walking Presentation

Main outline:

```text
G1_perceptive_walking_5min_outline.tex
```

Compile with XeLaTeX because the slides use Chinese text through `ctexbeamer`:

```bash
xelatex G1_perceptive_walking_5min_outline.tex
```

The deck includes report images under:

```text
assets/
```

Short presentation videos are under:

```text
assets/videos/
```

This makes the `presentation/` directory self-contained for Overleaf upload or local compilation.

For a 5-minute talk, use slides 1-9 as the main presentation and keep slide 10 as backup. The recommended emphasis is:

- Spend less time on generic reinforcement learning.
- Spend more time on reward loopholes, training collapse, and how the gait-shaping rewards fixed them.
- Use the result slides to show both quantitative learning curves and qualitative stepping/stair sequences.
