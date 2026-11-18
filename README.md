# Project plan: What can a protein–ligand GNN represent, and what can it not?
 
Status: revised 2026-11-17. Supersedes the 2026-11-13 plan, which framed this as a publication-oriented generalization study.
Amended 2026-11-18: §15 (code conventions) added; §6 Stage A and §13 updated to reference it.
 
Companion document: `claude/gnn_stage0_feasibility_guide.md` — now partly superseded, see §12.
 
---
 
## 0. Results you should be able to state at the end of reading this [Core]
 
1. The project is a **capability ladder**: a sequence of models, each adding exactly one representational capability over the previous one.
2. Between every rung there is a **diagnostic pass** that answers one question: which test examples does this model get wrong, and which property of the data explains the error?
3. You move up a rung only when you have a **documented failure mode** that the next rung's capability is supposed to fix. This is the rule that makes the iteration principled rather than random.
4. The starting point is a **plain GCN on 2D ligand graphs**, which has known, provable limitations. You will verify those limitations empirically before fixing them.
5. Chemoinformatics enters as the **slicing vocabulary** for error analysis: the set of chemical properties you use to describe which molecules the model fails on.
6. Every diagnostic must first be validated on a task with a known answer, otherwise a diagnostic that cannot detect a failure will report that there is none.
7. The evaluation split stays leakage-controlled (CleanSplit), so the numbers remain honest and comparable to published work. A paper is a possible by-product, not the goal.
8. All GNN code follows the torch_geometric API shape, written in plain torch. §15.
---
 
## 1. Definitions [Core]
 
Terms from the previous plan that still apply, restated in compressed form.
 
- **Ligand:** a small molecule, typically 20–60 heavy atoms, that binds a protein.
- **Binding affinity:** reported as $pK_d = -\log_{10} K_d$, with $K_d$ the dissociation constant in molar units. Higher means tighter binding. Range in these datasets: roughly $pK_d \in [2, 12]$.
- **PDBbind:** the standard dataset of protein–ligand complexes with measured affinities.
- **CleanSplit:** a leakage-reduced re-split of PDBbind, released with the GEMS paper (2025). It removes test complexes too similar to training complexes by both protein sequence similarity and ligand chemical similarity.
- **Data leakage:** a test example being a near-duplicate of a training example, so a model scores well by memorization.
- **Message passing:** for node $i$ at layer $l$,
  $$h_i^{(l+1)} = \phi\left(h_i^{(l)}, \; \bigoplus_{j \in \mathcal{N}(i)} \psi\left(h_i^{(l)}, h_j^{(l)}, e_{ij}\right)\right)$$
  with $\mathcal{N}(i)$ the neighbours of $i$, $e_{ij}$ the edge feature, $\bigoplus$ a permutation-invariant aggregation, and $\phi, \psi$ learned functions.
New terms introduced by this revision.
 
- **Expressive power:** which pairs of distinct graphs a model can assign different outputs to. A model that produces identical outputs for two graphs cannot predict different affinities for them, regardless of how it is trained.
- **1-WL (one-dimensional Weisfeiler–Leman) test:** an algorithm that iteratively recolours each node by hashing the pair (its own colour, the multiset of its neighbours' colours). Two graphs it assigns the same final colour multiset to are called 1-WL-indistinguishable. Standard message-passing GNNs cannot distinguish any pair that 1-WL cannot distinguish (Xu et al., ICLR 2019).
- **Error slice:** a subset of the test set defined by a property, for example "molecules with 4 or more rings". Reporting the error metric per slice rather than pooled.
- **Linear probe:** a linear regression fitted from a model's internal representation to a known property. If the probe cannot recover the property, the representation does not encode it linearly.
- **Murcko scaffold:** the ring systems of a molecule plus the linkers between them, with all side chains removed. Used to group molecules into chemical families.
- **Tanimoto similarity:** for two binary fingerprint vectors $a, b$, $T(a,b) = \dfrac{|a \cap b|}{|a \cup b|} \in [0,1]$. The standard chemical similarity measure.
- **Noise ceiling:** the lowest RMSE any model could achieve, set by the experimental uncertainty of the measured affinities themselves.
---
 
## 2. What changed from the previous plan, and why [Core]
 
| Aspect | Previous plan (2026-11-13) | This plan |
|---|---|---|
| Organizing question | Does the GNN use the protein, on an honest split? | What can each model represent, and which chemistry does it fail on? |
| Primary output | A publishable generalization study | Your understanding, plus an architecture improved by evidence |
| Starting model | Fingerprint + ridge regression (no GNN) | Plain GCN on 2D ligand graphs |
| Role of ablations | The novel contribution | One diagnostic among eight |
| Protein enters at | Stage 2, as the main subject | Rung L4, after ligand-side limits are exhausted |
| Chemoinformatics | Not addressed | The vocabulary for error slicing, developed alongside each rung |
| Publication | The goal | A possible by-product |
 
What is retained: CleanSplit as the evaluation split, the metric set, bootstrap confidence intervals, and the correctness-check discipline.
 
One consequence worth stating explicitly. The previous plan's novelty analysis (which ablations GEMS already published) now matters much less. You are no longer competing for novelty. You still check published results, but as reference points for your own numbers rather than as territory to avoid.
 
---
 
## 3. The capability ladder [Core]
 
Each rung adds one capability. The column "What it provably cannot do" is what you will test for at that rung.
 
| Rung | Model | Capability added | What it provably cannot do |
|---|---|---|---|
| L0 | Morgan fingerprint + ridge regression | None (no graph structure, no learning of structure) | Anything requiring the model to learn which substructures matter |
| L1 | **GCN**, 2D ligand graph, symmetric-normalized mean aggregation, no edge features | Learned local substructure detection | Distinguish bond orders; count rings or other substructures; distinguish 1-WL-equivalent graphs; see 3D shape; see the protein |
| L2 | GIN-style sum aggregation + bond-type edge features | Injective aggregation (1-WL-expressive); bond order | Count rings; distinguish 1-WL-equivalent graphs; see 3D; see the protein |
| L3 | L2 + explicit structural descriptors as node/graph features (ring membership, ring size, substructure counts) | Substructure counting, supplied rather than learned | Distinguish stereoisomers; see 3D conformation; see the protein |
| L4 | Heterogeneous graph: ligand atoms + pocket residues, relational message passing per edge type | The protein identity and the pocket composition | Use interatomic geometry; distinguish binding poses |
| L5 | L4 + interatomic distances as radial-basis edge features | Geometry, in a rotation- and translation-invariant form | Represent directional quantities such as vectors and orientations |
| L6 [Optional] | Equivariant GNN on 3D coordinates | Directional geometry | — |
 
### 3.1 The rule for moving up [Core]
 
Do not move from rung $k$ to rung $k+1$ until all three hold:
 
1. The model at rung $k$ is trained, and its pooled metrics are recorded with confidence intervals.
2. The diagnostic pass (§4) is complete, and you have written down at least one concrete failure mode in the form: "this model is worse on test examples with property $P$, and the mechanism I propose is $M$".
3. You can state why the capability added at rung $k+1$ addresses mechanism $M$, before you run it.
Condition 3 is a prediction. Record it before training. After training rung $k+1$, check whether the predicted error slice actually improved. This is the part that builds understanding, because a wrong prediction is more informative than a correct one.
 
### 3.2 Why the ladder starts at L1, not L0 [Core]
 
You chose a GCN-first start. L0 remains in the table because the code already exists in the Stage 0 guide, and it costs roughly one day. Run it as a background control, not as a gate. Its single purpose: if the GCN at L1 does not beat a linear model on fixed fingerprints, the GCN is either broken or learning nothing, and you must debug before interpreting anything.
 
---
 
## 4. The diagnostic pass [Core]
 
This is the core of the revised project. Run every item at every rung. Keep the outputs in one accumulating ledger (§7).
 
Define the per-example signed residual $e_i = \hat{y}_i - y_i$ in $pK_d$ units.
 
### 4.1 Diagnostic 1 — Slice the error by chemical property
 
Compute, for each test molecule, the properties in §5. For each property, bin the test set and report RMSE and Pearson $r$ per bin, with bootstrap intervals.
 
What to look for: a bin where the error is much larger than the pooled error, and where the bin is large enough that the interval excludes the pooled value.
 
### 4.2 Diagnostic 2 — Memorization versus generalization
 
1. For each test molecule, compute $s_i = \max_j T(\text{fp}_i, \text{fp}_j)$ over all training molecules $j$, with $T$ the Tanimoto similarity.
2. Plot $|e_i|$ against $s_i$ and report the Spearman correlation between them.
3. Report RMSE separately for $s_i < 0.4$, $0.4 \leq s_i < 0.6$, and $s_i \geq 0.6$.
Interpretation: a strong negative correlation between $|e_i|$ and $s_i$ means the model performs by interpolating between chemically similar training molecules. That is not useless, but it is a different claim than "the model learned binding".
 
Do the same with the protein: group test complexes by whether their protein family appears in training.
 
### 4.3 Diagnostic 3 — The expressiveness probe
 
This tests a provable limitation directly rather than inferring it.
 
1. Construct or find pairs of molecules $(A, B)$ in the dataset that are 1-WL-indistinguishable but have different measured affinity. Compute the 1-WL colouring yourself; it is about 20 lines of code and is a Tier 1 exercise that maps onto the CS224W expressiveness lecture.
2. Confirm the trained model gives $\hat{y}_A = \hat{y}_B$ up to numerical precision.
3. Report how many such pairs exist in the test set, and their contribution to total squared error.
If the count is near zero, this limitation is theoretically real but practically irrelevant for this dataset. That is itself a finding, and it tells you not to spend time on L2's injectivity fix for this reason alone. I am not certain how common such pairs are in PDBbind ligands; measuring it is the point.
 
### 4.4 Diagnostic 4 — Representation probes
 
1. Take the pooled graph embedding $z \in \mathbb{R}^d$ from the trained model, before the final prediction head.
2. For each chemical property $p$ in §5, fit a ridge regression from $z$ to $p$ on the training set and report $R^2$ on the test set.
3. A low $R^2$ means the model's representation does not encode $p$ linearly.
Interpretation requires care. A property the model does not encode is not necessarily a property it needs. Combine this with Diagnostic 1: a property that is both poorly encoded and associated with large error is a strong candidate for the next rung's fix.
 
### 4.5 Diagnostic 5 — Input ablation
 
At test time, and separately at training time:
 
1. Replace all node features with a constant. Measures how much comes from connectivity alone.
2. Replace the graph with an edgeless graph, keeping node features. Measures how much comes from atom composition alone, ignoring structure.
3. From rung L4: delete protein nodes; substitute a pocket from an unrelated complex; shuffle the protein–ligand pairing across the test set.
4. From rung L5: perturb coordinates by Gaussian noise with $\sigma \in \{0.5, 1.0, 2.0, 5.0\}$ Å; sweep the protein–ligand edge distance cutoff.
A model whose performance is unchanged by an ablation was not using that information.
 
In the conventions of §15, ablations 1 and 2 are edits to `data.x` and `data.edge_index` respectively, and ablation 3 is an edit to `data.edge_index` plus the node-type vector. Implement them as functions from `Data` to `Data`, so the same ablation code runs at every rung.
 
### 4.6 Diagnostic 6 — Attribution
 
Use integrated gradients with respect to atom features to obtain a per-atom attribution score. Compare the highest-scoring atoms against the atoms that form the chemical interactions you can identify in the structure (§5.3). Treat the comparison as qualitative evidence, not proof: attribution methods have known failure modes, and agreement with chemistry could be coincidental on a small number of examples.
 
### 4.7 Diagnostic 7 — The noise ceiling
 
Estimate the lowest RMSE achievable in principle. Measured affinities in PDBbind come from heterogeneous assays across many laboratories, so the same complex measured twice does not give the same number.
 
I am not certain of the correct value for PDBbind specifically. Published estimates of duplicate-measurement reproducibility in public affinity databases are of the order of 0.5 to 1.0 $pK$ units of standard deviation, which implies an RMSE floor of roughly that size. Find and cite a source before using a number. The practical consequence: once your RMSE approaches the ceiling, further architecture work is measuring noise.
 
### 4.8 Diagnostic 8 — Validate the diagnostics themselves [Core]
 
A diagnostic that cannot detect a failure you know is present will report that there is none. Before trusting any of the above on real data, run them on tasks with known answers.
 
1. **Known-answer task A:** train the model to predict the number of rings in a molecule. A mean-aggregation GCN should do measurably worse than a sum-aggregation GIN, because counting requires injective aggregation. If both succeed equally, your implementation of one of them is wrong.
2. **Known-answer task B:** train on a target that is a known function of a single atom feature, for example the count of nitrogen atoms. Every rung should solve this almost exactly. Failure means a pipeline bug.
3. **Known-answer task C:** train on a target that is pure noise, uncorrelated with the input. Every model should reach $r \approx 0$ on test. A model that does not is leaking the target.
Run these three before Diagnostic 1 at rung L1, and re-run A whenever you change the aggregation.
 
Task A is cheap under §15, because `GCNConv` and `GINConv` have the same call signature: the comparison is a one-line substitution with the rest of the pipeline held fixed, which is what makes the result interpretable.
 
---
 
## 5. Chemoinformatics component [Core]
 
Scope chosen: feature-level, developed in service of diagnostics. Each item below is something you must be able to compute with RDKit and explain in one sentence.
 
### 5.1 Per-atom and per-bond features
 
1. Element, degree, formal charge, total number of attached hydrogens, hybridization, aromaticity, ring membership.
2. Bond type (single, double, triple, aromatic), conjugation, ring membership, stereochemistry flag.
3. Partial charge (Gasteiger), as an example of a feature that depends on the whole molecule rather than the local neighbourhood.
Per §15, item 1 and 3 populate `data.x`; item 2 populates `data.edge_attr`, which is computed from L1 onward but consumed only from L2.
 
### 5.2 Per-molecule descriptors, used as slicing variables
 
1. Heavy atom count, molecular weight.
2. Number of rotatable bonds — a proxy for conformational flexibility, which a 2D model cannot see the consequences of.
3. Ring count, number of aromatic rings, largest ring size, fused-ring systems.
4. Fraction of sp3-hybridized carbons — a measure of three-dimensionality.
5. Calculated logP and topological polar surface area — proxies for hydrophobicity and polarity.
6. Hydrogen bond donor and acceptor counts.
7. Murcko scaffold, and the frequency of that scaffold in the training set.
8. Number of stereocentres, and whether stereochemistry is specified.
Slicing by item 2, 4 and 8 is the direct test of what a 2D graph model is missing relative to a 3D one. Slicing by item 7 is the direct test of scaffold memorization.
 
### 5.3 Interaction vocabulary, needed from rung L4
 
1. Hydrogen bond, with its distance and angle criteria.
2. Hydrophobic contact.
3. $\pi$-stacking, both parallel and T-shaped.
4. Salt bridge and other ionic interactions.
5. Halogen bond.
6. Buried surface area of the ligand on binding.
Compute these with an existing tool rather than implementing them. These form the chemical ground truth that Diagnostic 6's attributions are compared against.
 
### 5.4 Learning route
 
Do not study this as a separate block. Attach each item to the rung where it is first needed: §5.1 at L1, §5.2 at L1 for slicing, §5.3 at L4. One short written note per item, defining it and stating why it might matter for affinity.
 
---
 
## 6. Stages, with CS224W and MML mapping
 
Durations are estimates and should be adjusted to actual pace.
 
### Stage A — L1, the plain GCN (2–4 weeks) [Core]
 
Prerequisites: CS224W Lec 3–5 (message passing); MML Ch. 3 (linear algebra).
 
1. Build the 2D molecular graph from SMILES with RDKit, emitting a `Data` object per §15. Nodes are heavy atoms, edges are covalent bonds, each bond stored as two directed columns of `edge_index`.
2. Implement the GCN layer in plain PyTorch, as a `MessagePassing` subclass per §15:
   $$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$
   with $\hat{A} = A + I$ the adjacency matrix with self-loops and $\hat{D}$ its degree matrix. Derive the gradient with respect to $W^{(l)}$ by hand in index notation, then verify against autograd in float64. The backward is implemented as the transpose of `propagate`: `aggregate_backward` is a gather, `message_backward` is a scatter.
3. Pool with mean pooling to a graph vector, then a small multilayer perceptron to a scalar.
4. Run the three known-answer tasks of §4.8.
5. Run the OGB correctness check: train the same architecture on an OGB molecular property dataset with published numbers and confirm you land in the published range.
6. Train on CleanSplit. Record metrics.
7. Run the full diagnostic pass.
8. Write the failure-mode note required by §3.1.
Tier: Tier 1 — from scratch, until you can derive and code it without notes.
 
### Stage B — L2 and L3, fixing ligand-side representation (3–5 weeks) [Core]
 
Prerequisites: CS224W expressiveness lecture (1-WL, GIN).
 
1. Implement 1-WL yourself and run the expressiveness probe of §4.3.
2. Implement sum aggregation with an MLP, and edge-feature conditioning. Under §15 this is a `GINConv` with `aggr='add'` and a `message` that consumes `edge_attr`; the pipeline around it does not change.
3. Predict, before training, which error slice should improve. Train. Check.
4. Add explicit structural descriptors at L3. Predict, train, check.
### Stage C — L4, the protein (4–6 weeks) [Core]
 
Prerequisites: CS224W heterogeneous-graph and relational-GNN lectures.
 
1. Two node types: ligand atoms, pocket residues within a cutoff (start at 8 Å).
2. Three edge types: ligand–ligand bonds, protein–protein adjacency, protein–ligand proximity. Store the type as an `edge_type` vector of length $E$ alongside `edge_index`, per §15.
3. Separate message function $\psi_r$ per edge type $r$, selected inside `message` by masking on `edge_type`.
4. Reproduce the published protein-deletion gap as a pipeline check: GEMS reports CASF-2016 RMSE 1.308 with protein nodes and 1.572 without, both trained on CleanSplit. I transcribed these from the preprint text by automated read, not from the published tables; verify before relying on them.
5. Full diagnostic pass, now including the protein-side slices and ablations.
6. Retire the dense test oracle of §15.4 at this point, and record in the ledger which degree convention (in-degree, out-degree, or symmetrized) the normalization uses, since the two stop coinciding here.
### Stage D — L5, geometry (3–4 weeks) [Core]
 
Prerequisites: MML Ch. 5 (vector calculus), for gradients through distance computations.
 
1. Add $d_{ij} = \lVert x_i - x_j \rVert_2$ expanded in a radial basis:
   $$e_{ij}^{(k)} = \exp\left(-\gamma\left(d_{ij} - \mu_k\right)^2\right), \quad k = 1,\dots,K$$
   with centres $\mu_k$ evenly spaced over the cutoff range. These are $(E, K)$ entries of `edge_attr`, not an $(N, N, K)$ tensor; see §15.3.
2. Distances are invariant under rotation and translation, so this supplies geometric information without an equivariant architecture.
3. Run the coordinate-noise and cutoff-sweep ablations.
### Stage E — L6, equivariance (optional) [Optional]
 
Only if Stage D shows that geometry matters and that invariant distances are insufficient. Requires more mathematical background than you currently have. Revisit after MML Ch. 4 (matrix decompositions).
 
### Stage F — Consolidation (2–3 weeks) [Core]
 
Prerequisites: MML Ch. 6 (probability).
 
1. Bootstrap intervals for everything in the ledger.
2. A single figure: pooled metric against rung, with the per-slice breakdown underneath.
3. A written account of which predicted improvements happened and which did not.
4. Decide at this point whether there is a paper. §9.
---
 
## 7. The results ledger [Core]
 
Maintain one table, appended to after every rung. This is the project's actual output.
 
| Rung | Pooled $r$ | Pooled RMSE | Worst slice | Slice RMSE | Memorization $\rho$ | Predicted fix | Did it work? |
|---|---|---|---|---|---|---|---|
| L0 | | | | | | n/a | n/a |
| L1 | | | | | | | |
| L2 | | | | | | | |
 
Columns, defined:
 
1. **Worst slice:** the property bin with the largest RMSE excess over pooled.
2. **Memorization $\rho$:** the Spearman correlation between $|e_i|$ and nearest-training-neighbour Tanimoto similarity, from §4.2.
3. **Predicted fix:** what you wrote down before training the next rung.
4. **Did it work?:** yes, no, or partly, with one sentence.
Record alongside the table the conventions listed in §15.5, each time one of them changes.
 
---
 
## 8. Metrics and reporting [Core]
 
Unchanged from the previous plan.
 
1. Pearson correlation $r$.
2. Spearman rank correlation $\rho$.
3. RMSE in $pK_d$ units.
4. Per-slice breakdowns, not only the pooled number.
5. Bootstrap percentile confidence intervals, 1,000 resamples, for every reported figure.
Reference points on CASF-2016 ($N = 285$), from the GEMS preprint, all transcribed by automated read and requiring verification: GEMS full complex model $r = 0.803$, RMSE 1.308; GenScore $r = 0.780$; Pafnucy $r = 0.746$; top-5 ligand similarity search $r = 0.625$, RMSE 1.711.
 
---
 
## 9. Publication, as a by-product
 
Reconsider at Stage F, not before. If the ledger contains a failure mode that is general rather than specific to your implementation, and the diagnostic that found it is reusable, that is a methods contribution. Candidate venues remain the machine-learning-for-structural-biology workshops at NeurIPS or ICML, or the Journal of Chemical Information and Modeling.
 
Do not let this influence which experiments you run. The previous plan's novelty constraints are withdrawn.
 
---
 
## 10. Compute [Core]
 
1. MacBook Air, 16 GB RAM: sufficient for L0, for L1 on subsets, for all RDKit descriptor computation, and for every diagnostic except full training runs.
2. MPCDF Raven A100 partition: full training at all rungs. Models of this size train in hours, not days.
3. Avoid co-folding models of the AlphaFold-3 type and any pretraining run on millions of molecules. Both convert a tractable project into an intractable one.
Note on the diagnostic workload: the diagnostics are cheap relative to training. Most run on CPU in minutes from saved predictions. Save per-example predictions for every trained model so diagnostics can be re-run without retraining.
 
---
 
## 11. Risks
 
| Risk | Mitigation |
|---|---|
| A broken model produces a confident false diagnosis | The three known-answer tasks of §4.8 plus the OGB check at L1, re-run after every architecture change |
| Error slicing finds spurious patterns because many slices are tested | Decide the slice list in advance, report the number of slices tested, treat an isolated significant slice among many as a hypothesis rather than a result |
| Attribution results over-interpreted | §4.6 is explicitly qualitative; no conclusion rests on it alone |
| Chasing improvements below the noise ceiling | Estimate the ceiling at §4.7 before Stage C, and stop when within it |
| Scope creep into equivariance | Stage E is optional and gated on Stage D's outcome |
| PDBbind licensing restricts redistribution | Distribute code and split indices, not data; check terms before publishing anything derived |
| A batching bug silently connects two molecules | §15.4 check 4 against the dense oracle, plus the `edge_index` bounds assertion of §15.2 |
 
---
 
## 12. Status of the companion Stage 0 guide
 
`claude/gnn_stage0_feasibility_guide.md` was written for the previous framing, where a ligand-only ridge baseline was a go/no-go feasibility gate. Under this plan it is no longer a gate.
 
Still useful, unchanged:
 
1. The data-acquisition section: where CleanSplit lives, the Zenodo datasets, and the three paths if the precomputed graphs lack SMILES.
2. The split safety checks.
3. The B1 ridge code, now serving as the L0 control of §3.2.
4. The reference-number table and its cautions.
No longer applicable: the decision checkpoint table, which decided whether to proceed and with what question. You proceed regardless, and the question is fixed by this plan.
 
---
 
## 13. Immediate next actions
 
1. Resolve data access: clone the GEMS repository, inspect `PDBbind_data/PDBbind_data_split_cleansplit.json`, and determine whether ligand SMILES are obtainable from the Zenodo files or require PDBbind registration.
2. Run the L0 ridge control using the existing code. One day, no interpretation beyond "the pipeline works".
3. Build the 2D graph construction from SMILES with the §5.1 features, emitting `Data(x, edge_index, edge_attr)` per §15. **Done 2026-11-18.**
4. Implement the GCN layer as a `MessagePassing` subclass, with the hand-derived gradient verified against autograd in float64 and against finite differences. **Done 2026-11-18.**
5. Run the three known-answer tasks of §4.8 before touching CleanSplit.
---
 
## 14. Checkpoints, with answers [Core]
 
**Q1.** Why does the ladder put sum aggregation at L2 rather than using it from the start?
 
**A1.** Because the project's purpose is to observe the limitation before fixing it. A mean-aggregation GCN cannot distinguish two neighbourhoods with the same distribution of neighbour features but different counts, for example one nitrogen neighbour versus two nitrogen neighbours in the same proportion of a larger neighbourhood. Starting with sum aggregation would skip the observation and leave you with a working model you do not understand.
 
**Q2.** You find that test RMSE is much higher for molecules with more than six rotatable bonds. Name two mechanisms, and say how to distinguish them.
 
**A2.** Mechanism 1: conformational flexibility costs binding entropy, which a 2D graph cannot represent, so the model cannot learn the penalty. Mechanism 2: flexible molecules are larger on average, and large molecules are rarer in the training set, so the error reflects data scarcity rather than representation. Distinguish them by re-slicing on rotatable bonds while holding heavy atom count fixed, that is, within narrow size bins. If the effect survives within a size bin, mechanism 1 is supported.
 
**Q3.** The linear probe recovers ring count from the L1 embedding with $R^2 = 0.9$. Does this contradict the claim that message passing cannot count rings?
 
**A3.** No, for two reasons. First, the theoretical statement is that 1-WL-bounded message passing cannot count cycles in general, meaning there exist graph pairs it cannot distinguish by cycle count; it does not say the count is never recoverable on a particular dataset. Second, on real molecules, ring count correlates strongly with other quantities the model does encode, such as heavy atom count and aromaticity fractions, so the probe may be recovering a correlate rather than the count itself. Test this by probing on a set where ring count is decorrelated from size.
 
**Q4.** Your L2 model improves pooled RMSE by 0.04, with a bootstrap interval of $[-0.02, 0.10]$. What do you record in the ledger?
 
**A4.** Record the improvement as not established, because the interval contains zero. Then check the specific slice you predicted would improve. A clear improvement in the predicted slice, with no pooled change, is a meaningful and recordable outcome: the fix did what you expected, but that slice is too small to move the pooled number.
 
---
 
## 15. Code conventions [Core]
 
Added 2026-11-18. These apply to every rung and to every diagnostic.
 
### 15.1 The rule
 
All GNN code follows the **torch_geometric API shape**, implemented in **plain torch**. The library itself is imported only for data loading and for reference comparisons in tests; it never appears in the model.
 
The distinction:
 
- **PyG style** is the data layout and the API decomposition: `Data`, `Batch`, `MessagePassing`, `propagate` → `message` → `aggregate` → `update`, `edge_index`, `edge_attr`, `batch`, `global_mean_pool`. This is a convention and carries no dependency.
- **PyG library** is importing `torch_geometric.nn` layers and using them as the model. Not permitted here, because the point of Tier 1 is that the layer can be derived and written from nothing.
Rationale: the API shape is what makes L1 → L2 → L4 a change of two methods rather than a rewrite, it matches how the course and the literature describe message passing, and it keeps the code comparable to published implementations. Writing the layers yourself is what makes the derivation load-bearing.
 
### 15.2 Layout and naming
 
1. `data.x` — node features, $(N, F)$. §5.1 items 1 and 3.
2. `data.edge_index` — $(2, E)$, `dtype=torch.long`. `edge_index[0] = j` is the source, `edge_index[1] = i` is the target; messages flow $j \to i$. Every undirected bond is stored as **two** columns.
3. `data.edge_attr` — $(E, F_e)$, aligned column-for-column with `edge_index`. §5.1 item 2. Computed from L1, consumed from L2.
4. `data.edge_type` — $(E,)$ integer, from L4.
5. `data.y` — the target, $(1,)$ per graph.
6. `batch.batch` — $(N,)$ integer, assigning each node to its graph. Batching is a disjoint union with offset edge indices, never padding.
7. Assert `batch.edge_index.max() < batch.x.shape[0]` after every collation. A missed batch offset otherwise fails silently by connecting one molecule's last atom to the next molecule's first.
### 15.3 Layer structure
 
1. Every layer subclasses `MessagePassing` and overrides `message`, and `update` where needed. Nothing else.
2. A layer applies **no activation**; the caller applies it. This matches PyG and keeps the ReLU derivative with the caller in the manual-gradient exercise.
3. Per-edge quantities — the GCN normalization, bond features, radial basis expansions — are $(E, \cdot)$ tensors passed through `propagate` as keyword arguments, never $(N, N, \cdot)$ tensors. At L5 with $K=32$, $N=400$, $B=64$ in float32, the dense form is 1.31 GB against 10.1 MB for the edge-list form.
4. Manual backward passes, where the exercise calls for one, are written as the transpose of `propagate`: `aggregate_backward` is a gather, `message_backward` is a scatter. Scatter and gather are transposes of each other, which is the computational content of $\partial L/\partial H = S^\top \Delta W^\top$.
### 15.4 The test suite
 
Six checks, run after every architecture change. The first four are the ones with independent content.
 
1. The normalized adjacency $S$ matches a value computed by hand on a 3-node path.
2. The manual backward matches autograd, in float64.
3. The manual backward matches central finite differences, in float64. This is the only check independent of PyTorch's own backward.
4. The layer matches the dense formula $SXW + b$, computed with a padded `to_dense` oracle. This is the only check independent of the PyG API itself.
5. The layer matches `torch_geometric.nn.GCNConv` with parameters copied across, forward and gradient. This verifies transcription, not understanding: it passes even if the algorithm was copied without being derived.
6. The graph-level output is invariant to a node permutation, applied as `x[p]` together with `inv[edge_index]` for the inverse permutation `inv`.
Checks 1 and 4 depend on the dense oracle, which has no analogue once the graph is heterogeneous. Retire them at L4; keep 2, 3, 5 and 6.
 
Float64 for all gradient checks. Float32 for training runs only, with checks 2 and 3 re-run in float64 after any change.
 
### 15.5 Conventions to record in the ledger
 
Each of these is a free choice that will silently change results if it changes between rungs.
 
1. Whether the linear map is applied before or after aggregation. These agree while `message` is linear in `x_j` and stop agreeing at L2 with edge conditioning.
2. Which index the degree is computed from. In-degree and out-degree coincide for undirected graphs with both directions stored, and stop coinciding at L4.
3. The pooling function. Mean pooling is size invariant; sum pooling is not, and lets the head read molecule size directly.
4. Whether self-loops are added.
5. The floating-point precision of the run.
---
 
## Sources
 
- GEMS, Nature Machine Intelligence, 2025: https://www.nature.com/articles/s42256-025-01124-5
- GEMS preprint (full text): https://www.biorxiv.org/content/10.1101/2024.12.09.627482v2.full
- GEMS code (MIT licensed): https://github.com/camlab-ethz/GEMS
- GEMS precomputed datasets: https://doi.org/10.5281/zenodo.15482796
- Xu et al., How Powerful are Graph Neural Networks? (GIN, 1-WL bound), ICLR 2019
- Chen et al., Can Graph Neural Networks Count Substructures?, NeurIPS 2020
- Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017
- Fey and Lenssen, Fast Graph Representation Learning with PyTorch Geometric, ICLR 2019 workshop (the API this project's conventions follow)
