# Checkpoints

## Release Checkpoint
- **File**: `checkpoints/slither_ml_brain_gen100.json`
- **Role**: Release landmark (Century Apex Champion).
- **Lineage Details**:
  - Generations: 100
  - Peak Fitness: 43,634
  - Best Fitness (Floor): 37,089 (85% floor)
  - Lessons Learned: 1.08M
  - Kills: 168
  - Peak Length: 22,810

This checkpoint represents the stabilized landmark model referenced in `console_inject.js` provenance. Generations 100 through 337 share identical champion weights due to the stagnation floor.

## Historical Archive
All other training generation checkpoints (generations 8 through 536) have been moved out of version control into:
- `checkpoints_archive/`

This keeps git clones and repository history lightweight while preserving all historical lineages locally for offline benchmarking and evolutionary diagnostics.
