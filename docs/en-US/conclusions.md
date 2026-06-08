# Current Conclusions

## Research Status

**Original hypothesis:** MultiState neurons are intrinsically superior to traditional neurons.  
**Status:** refuted.

**Revised hypothesis:** compact experts with Top-1 sparse routing can preserve dense-network performance while reducing inference compute.  
**Status:** partially supported on small datasets.

**Most honest formulation:** the hypothesis of intrinsic MultiState neuron superiority was refuted; the narrower hypothesis of efficient conditional computation through sparse experts received positive experimental evidence on small problems.

---

✅ **States learn different representations:** Correlation between states is near zero. They are independent experts.

✅ **The Gate strongly influences the outcome:** Softmax routing successfully distributes workload in the first layer but suffers from mode collapse in deeper layers.

✅ **Sparse Routing reduces FLOPs:** By using Top-1 Hard Routing (V4), inference computational cost is reduced by approximately 50% compared to a dense network.

⚠️ **No evidence of absolute accuracy gain:** The architecture matches the traditional MLP, but does not surpass its upper bound of intelligence on the tested datasets.

⚠️ **Load Balancing struggles in shallow networks:** The MoE auxiliary loss did not improve the mode collapse in Layer 2, likely due to the limited depth and capacity of the network.
