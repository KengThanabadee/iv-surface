# implied-vol-surface

`implied-vol-surface` is a Deribit-backed crypto-options market data and
analytics workspace in development. The active path begins with a normalized,
provenance-rich BTC inverse-option snapshot and builds analytical components
from that data contract.

## Active direction

The first vertical slice is:

```text
Deribit source snapshot
        -> normalized long-form option records
        -> matched Call/Put Composite IV
        -> heatmap, smiles, and ATM term structure
```

Deribit `mark_iv` will be treated as provider-supplied reference IV. Bid/ask IV,
quote state, Call/Put gaps, timestamps, and source fields remain available as
quality and provenance context.

The normalized data contract is designed before chart APIs so the Deribit
source schema defines the foundation.

## Development

Use the repository environment:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

## History

Earlier Bybit price-to-IV and grid-first visualization experiments remain
recorded in the `bybit-solved-iv-study` and `grid-visualization-study` Git tags
as part of the project's learning history.
