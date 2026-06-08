# Research Diary

## 2026-06-08

### Experiment 1: Baseline Hypothesis Check (Test 12)
**Hypothesis:** Independent internal states can exist without collapsing into a single linear transformation.
**Result:** Confirmed.
**Observation:** Correlation between states is close to zero (~0.01). The states do not become identical copies of each other.

---

### Experiment 2: State Specialization (V2 & V3)
**Hypothesis:** An input-dependent Gate can specialize the usage of these internal states.
**Result:** Confirmed.
**Observation:** In Layer 1, the gate successfully routes different inputs to different states. However, in Layer 2, we observed *Mode Collapse*, where a single state handles ~80% of the traffic.

---

### Experiment 3: Sparse Routing & Noise Ablation (V4)
**Hypothesis:** The Softmax gate leaks noise from "bad" states. Using a Top-1 Sparse Routing will prevent this leakage.
**Result:** Confirmed.
**Observation:** Removing state 1 (L1_E1) in V4 has 0.00pp impact, proving it was perfectly gated out and produced no noise. V4 successfully eliminates the noise leakage observed in V3.

---

### Experiment 4: 30-Seed Validation (V4 vs Traditional)
**Hypothesis:** The V4 architecture can match the traditional MLP performance.
**Result:** Confirmed.
**Observation:** 
- Traditional MLP: 87.84% ± 1.36%
- V4 Sparse: 87.67% ± 1.15%
**Conclusion:** Technical tie in accuracy. However, V4 achieves this using approximately 50% of the FLOPs during inference, validating that sparse routing and internal experts can drastically reduce computational cost while maintaining the same intelligence.

---

### Experiment 5: Stress Testing (Moons, Spirals, 20-Features)
**Hypothesis:** The efficiency discovered in V4 holds across complex domains.
**Result:** Confirmed.
**Conclusion:** V4 matched the Traditional MLP across all tested complex datasets, solidifying the architecture's validity.

---

### Phase 1 Synthesis: Hypothesis Reformulation
**Research status:** Promising, but not yet scaled.

**Original hypothesis:** MultiState neurons are intrinsically superior to traditional neurons.
**Result:** Refuted.

**Revised hypothesis:** Compact experts with Top-1 sparse routing can reproduce the performance of dense networks while using less inference compute.
**Result:** Partially supported on small datasets.

**Accumulated evidence:**
- V1 failed: linear state aggregation collapses into an effectively dense transformation.
- V2 partially failed: the Softmax gate specializes in the first layer, but leaks noise.
- V3 approached the baseline: MLP gate + skip connections recovered performance.
- V4 reached a technical tie: Top-1 Sparse Routing eliminated leakage from bad states.
- 30-seed validation reduced the risk that the result came from a favorable seed.
- Stress tests on Moons, Spirals, and 20-feature data preserved the technical tie.

**Current conclusion:** V4 has not shown absolute accuracy superiority. The positive result is efficiency: similar performance to a traditional MLP with roughly half the inference FLOPs.

**Main risk:** All experiments are still small, with low dimensionality and few classes. There is not enough evidence yet to claim scalability to real computer vision or Transformers.

**Next critical milestone:** MNIST.
