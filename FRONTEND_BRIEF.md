# Frontend brief (build this as a real website)

Goal: a multi-section website that explains the whole project to a professor and a lay reader: problem, how the simulator works, the experiments, the findings, the limits. Plus a live simulation. Deliver as a static site in `site/` (plain HTML/CSS/JS, no build step), publish it as an artifact, and add it to the zip.

## Data you already have
- `dashboard/web/data.js` / `dashboard/export_data.py`: all results, scripted baselines, learning curves, recorded episodes.
- `results/summary.md`, `PROGRESS.md`, `paper/draft.md`: the honest findings and caveats. Report failed hypotheses as failed.
- `agrinegotiate/scenarios.py`, `agrinegotiate/agents/scripted.py`: rules to port to JS for the live simulation.

## Pages (one scrolling site with a sticky side index, or separate pages)
1. **The problem**: a smallholder farmer, a buyer, a price, a poor phone signal. One paragraph and one simple drawing. Who it helps and why a benchmark was needed.
2. **How a negotiation works**: step-by-step diagram of one round (offer, accept, walk, wait; price grid; deadline; delay). Show the gains-from-trade split with the worked example (800 / 1300, Nash price, farmer share).
3. **Simulator (live, in the browser)**: user picks scenario, farmer and buyer strategy (scripted ones ported to JS), link quality and platform rule (none / fee / floor). Animate offers on a price line between the two reservation values, with a log. Also let the user play the buyer or farmer against a scripted agent. Replay recorded learned-agent episodes from data.js.
4. **How we measured**: the metrics (efficiency, farmer share, agreement, Nash distance), the seeds and intervals, in plain words, with the yardsticks drawn.
5. **Findings**: one block per hypothesis H1 to H5 with verdict (held / failed / mixed), the chart, the number and the sample size. Lead with: the fee helps farmers but a plain price floor beats it in the main test. Mark the buyer-variety result as exploratory.
6. **Limits and next steps**: copy from PROGRESS.md.

## Design rules (must not look AI generated)
- No emoji, no gradient blobs, no glassmorphism, no purple, no identical rounded cards in a grid, no big stat-tile row, no "hero with two buttons".
- Editorial almanac feel: a strong serif for headings, a readable text face, generous margins, a narrow reading column with wide figures that break out of it, margin notes, thin rules, small numbered figures with captions ("Fig. 3").
- Palette from the field: unbleached paper, soil brown ink, one wheat-gold accent, one leaf green for "good", one clay red for "bad". Dark mode optional.
- Drawn with SVG, not stock icons: a simple horizon with furrows, hand-set price line, a small farmer and buyer mark made of plain shapes.
- Every chart has a sentence above it saying what to notice. Honest uncertainty shown (dots per run, intervals).
- Mobile friendly. Fast. Fonts from Google Fonts with system fallbacks.

## Budget and stopping
- Do the work in few, large tool calls: write files with one Write each, screenshot with headless Chrome (`google-chrome --headless --no-sandbox --screenshot`) once per page, fix what is wrong, stop.
- No new experiments, no refactors, no extra research. If usage is running high, finish the simulator and findings first, then the rest, and stop.
- When done: publish to the existing artifact slug 2c9b8ab6-3219-4ba5-9ea4-197173e6c02f (list_files first, base_version from get_artifact), re-zip to /home/claude/agrinegotiate.zip, SendUserFile the zip, and write 5 lines to PROGRESS.md.
