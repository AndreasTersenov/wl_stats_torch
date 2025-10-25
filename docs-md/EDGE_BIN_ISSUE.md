# Edge Bin Issue in L1-Norm Calculation

## Problem Identified

The L1-norm calculation in `wl_stats_torch` includes values outside the specified `[min_snr, max_snr]` range by accumulating them in the **edge bins**, while `cosmostat` **excludes** them entirely.

## Root Cause

**File**: `wl_stats_torch/statistics.py`, function `compute_wavelet_l1_norms()`

**Lines 255-256**:
```python
bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)
bin_indices = torch.clamp(bin_indices, 1, n_bins)  # ← THE PROBLEM
```

The `torch.clamp()` operation forces all bin indices into the range `[1, n_bins]`:
- SNR values **< min_snr** → bin_indices clamped to 1 → accumulated in **first bin**
- SNR values **> max_snr** → bin_indices clamped to n_bins → accumulated in **last bin**

## Comparison with CosmoStat

**CosmoStat behavior** (`hos_peaks_l1.py` lines 648-651):
```python
digitized = np.digitize(ScaleSNR, thresholds_snr)
bin_l1_norm = [
    np.sum(np.abs(ScaleSNR[digitized == j]))
    for j in range(1, len(thresholds_snr))  # Only bins 1 to n_bins
]
```

When `np.digitize()` encounters values outside the range:
- Values < min_snr → bin index 0 (excluded from loop)
- Values > max_snr → bin index `n_bins+1` (excluded from loop)

## Evidence from Testing

Test with `min_snr=-10`, `max_snr=10` on Scale 0:

```
Cosmostat: 60 values < -10.0, 45 values > 10.0
wl_stats:  62 values < -10.0, 52 values > 10.0

L1-norm in FIRST bin:
  Cosmostat: 57.27
  wl_stats:  1259.52  ← 22x larger! (includes all values < -10)

L1-norm in LAST bin:
  Cosmostat: 75.63
  wl_stats:  1017.01  ← 13x larger! (includes all values > 10)

Sum of |SNR| outside range: 2134.65
Sum of L1 in edge bins:     2276.53  ← Almost exactly matches!
Total L1-norm difference:   2135.03  ← Confirms edge bins hold excluded values
```

## Impact

**When using data-dependent ranges** (`min_snr=None`, `max_snr=None`):
- Both implementations use actual min/max from data
- All values fall within range
- **No impact** - results match well

**When using fixed ranges** (e.g., `min_snr=-10`, `max_snr=10`):
- Values outside range get binned differently
- Edge bins in wl_stats_torch can have 10-20x more power
- **Significant impact** - results differ substantially

## Implications

### Current Behavior (with clamp):
**Pros:**
- ✅ No data is lost - total L1-norm includes all signal
- ✅ Edge bins show "how much power is beyond my limits"
- ✅ Useful for diagnostic purposes

**Cons:**
- ❌ Inconsistent with cosmostat reference implementation
- ❌ Edge bins are misleading (don't represent SNR at bin center)
- ❌ Can't directly compare with cosmostat results

### Proposed Behavior (without clamp):
**Pros:**
- ✅ Matches cosmostat exactly
- ✅ Edge bins represent true SNR at bin center
- ✅ Direct comparison with literature/other codes

**Cons:**
- ❌ Data outside range is silently dropped
- ❌ Total L1-norm no longer conserved if range is restrictive

## Recommendation

**Option 1: Match cosmostat (recommended)**
- Remove the `clamp()` operation
- Only process bins 1 through n_bins
- Values outside range are excluded (matching cosmostat)

**Option 2: Make it configurable**
- Add parameter `include_edge_overflow=False`
- When `True`: keep current behavior (clamp)
- When `False`: match cosmostat (no clamp)

**Option 3: Keep current behavior but document**
- Add clear warning in docstring
- Note that edge bins include overflow
- Users should be aware when comparing to cosmostat

## Recommended Fix

Replace lines 255-256 in `statistics.py`:

```python
# OLD (current):
bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)
bin_indices = torch.clamp(bin_indices, 1, n_bins)

# NEW (matches cosmostat):
bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)
# No clamp - values outside [min_snr, max_snr] get bin indices 0 or n_bins+1
# These are automatically excluded by the loop range(1, n_bins+1)
```

This would make the behavior identical to cosmostat.

## Test Results After Fix (Expected)

With the fix, when using `min_snr=-10`, `max_snr=10`:
- Edge bins should match cosmostat values
- Total L1-norm will be lower (excluding out-of-range values)
- Per-bin values should match within ~1-3% (due to wavelet coefficient precision)
