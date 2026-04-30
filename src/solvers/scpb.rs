use satcoder::{Bool, SatInstance};

/// Sequential Counter for Pseudo-Boolean constraints (SCPB).
///
/// Encodes the constraint `sum_{i=1}^{n} w_i * X_i <op> k` where <op> is <=, >=, or =.
///
/// Uses counter variables R[i][j] where R[i][j] = true iff
/// the weighted sum of the first (i+1) inputs is >= (j+1).
///
/// The key property for incremental solving: the bound k can be controlled
/// via SAT solver assumptions on R[n-1][k], without adding/removing clauses.
#[derive(Debug)]
pub struct SCPB<L: satcoder::Lit> {
    /// Input literals X_1, ..., X_n
    inputs: Vec<Bool<L>>,
    /// Weights w_1, ..., w_n
    weights: Vec<usize>,
    /// Counter variables R[i][j], 0-indexed.
    /// R[i] has pos_i entries where pos_i = min(k_max, cumsum of weights up to i)
    /// R[i][j] = true iff sum of first (i+1) weighted inputs >= (j+1)
    r_vars: Vec<Vec<Bool<L>>>,
    /// Maximum bound currently supported
    k_max: usize,
    /// Cumulative sums: cumsum[i] = sum of weights[0..=i]
    cumsums: Vec<usize>,
}

impl<L: satcoder::Lit + Copy + std::fmt::Debug> SCPB<L> {
    /// Create a new SCPB encoding.
    ///
    /// Encodes the weighted sum `sum w_i * X_i` with counter variables R
    /// that allow querying and constraining the sum via assumptions.
    ///
    /// # Arguments
    /// * `solver` - SAT solver instance
    /// * `inputs` - Boolean variables X_1, ..., X_n
    /// * `weights` - Corresponding weights w_1, ..., w_n (each >= 1)
    /// * `k_max` - Maximum bound to encode (controls size of R variables)
    pub fn new<S: SatInstance<L>>(
        solver: &mut S,
        inputs: Vec<Bool<L>>,
        weights: Vec<usize>,
        k_max: usize,
    ) -> Self {
        assert_eq!(inputs.len(), weights.len());
        let n = inputs.len();

        // Compute cumulative sums
        let mut cumsums = Vec::with_capacity(n);
        let mut cumsum = 0usize;
        for &w in &weights {
            cumsum += w;
            cumsums.push(cumsum);
        }

        // Create R variables
        let mut r_vars: Vec<Vec<Bool<L>>> = Vec::with_capacity(n);
        for i in 0..n {
            let pos_i = k_max.min(cumsums[i]);
            let r_i: Vec<Bool<L>> = (0..pos_i).map(|_| solver.new_var()).collect();
            r_vars.push(r_i);
        }

        let mut scpb = Self {
            inputs,
            weights,
            r_vars,
            k_max,
            cumsums,
        };

        // Add encoding clauses for all inputs
        scpb.add_base_clauses(solver, 0, n);
        scpb
    }

    /// Create an empty SCPB that can be extended incrementally.
    pub fn new_empty(k_max: usize) -> Self {
        Self {
            inputs: Vec::new(),
            weights: Vec::new(),
            r_vars: Vec::new(),
            k_max,
            cumsums: Vec::new(),
        }
    }

    /// Add base encoding clauses (1)-(6) for inputs in range [from, to).
    fn add_base_clauses<S: SatInstance<L>>(&mut self, solver: &mut S, from: usize, to: usize) {
        for i in from..to {
            let w_i = self.weights[i];
            let pos_i = self.r_vars[i].len();

            // Formula (1): X_i -> R_{i,j} for j in 1..=w_i
            // (0-indexed: j in 0..w_i)
            for j in 0..w_i.min(pos_i) {
                solver.add_clause(vec![!self.inputs[i], self.r_vars[i][j]]);
            }

            if i > 0 {
                let pos_prev = self.r_vars[i - 1].len();

                // Formula (2): R_{i-1,j} -> R_{i,j}
                for j in 0..pos_prev.min(pos_i) {
                    solver.add_clause(vec![!self.r_vars[i - 1][j], self.r_vars[i][j]]);
                }

                // Formula (3): X_i AND R_{i-1,j} -> R_{i,j+w_i}
                for j in 0..pos_prev {
                    let target = j + w_i;
                    if target < pos_i {
                        solver.add_clause(vec![
                            !self.inputs[i],
                            !self.r_vars[i - 1][j],
                            self.r_vars[i][target],
                        ]);
                    }
                }

                // Formula (4): !X_i AND !R_{i-1,j} -> !R_{i,j}
                // Equivalent to: R_{i,j} -> X_i OR R_{i-1,j}
                for j in 0..pos_prev.min(pos_i) {
                    solver.add_clause(vec![
                        !self.r_vars[i][j],
                        self.inputs[i],
                        self.r_vars[i - 1][j],
                    ]);
                }

                // Formula (6): !R_{i-1,j} -> !R_{i,j+w_i}
                // Equivalent to: R_{i,j+w_i} -> R_{i-1,j}
                for j in 0..pos_prev {
                    let target = j + w_i;
                    if target < pos_i {
                        solver.add_clause(vec![
                            !self.r_vars[i][target],
                            self.r_vars[i - 1][j],
                        ]);
                    }
                }
            }

            // Formula (5): !X_i -> !R_{i,j} for j in (pos_{i-1}+1)..=pos_i
            // (0-indexed: j in pos_prev..pos_i)
            // Only when pos_{i-1} < k
            let pos_prev = if i > 0 { self.r_vars[i - 1].len() } else { 0 };
            if pos_prev < self.k_max {
                for j in pos_prev..pos_i {
                    // R_{i,j} -> X_i
                    solver.add_clause(vec![!self.r_vars[i][j], self.inputs[i]]);
                }
            }
        }
    }

    /// Add AMK clause (8) for inputs in range [from, to).
    /// Formula (8): X_i -> !R_{i-1, k+1-w_i}  (overflow prevention)
    fn add_amk_clauses<S: SatInstance<L>>(&self, solver: &mut S, k: usize, from: usize, to: usize) {
        for i in from.max(1)..to {
            let w_i = self.weights[i];
            let pos_prev = self.r_vars[i - 1].len();
            // k+1-w_i in 1-indexed = k-w_i in 0-indexed (since k+1-w_i-1 = k-w_i)
            if k >= w_i {
                let idx = k - w_i; // 0-indexed
                if idx < pos_prev {
                    solver.add_clause(vec![!self.inputs[i], !self.r_vars[i - 1][idx]]);
                }
            }
        }
    }

    /// Incrementally add a new input X_{n+1} with weight w_{n+1}.
    ///
    /// This extends the SCPB encoding by creating R_n variables and
    /// adding clauses (1)-(6) for the new input.
    pub fn add_input<S: SatInstance<L>>(
        &mut self,
        solver: &mut S,
        input: Bool<L>,
        weight: usize,
    ) {
        let n = self.inputs.len();
        let prev_cumsum = if n > 0 { self.cumsums[n - 1] } else { 0 };
        let new_cumsum = prev_cumsum + weight;

        self.inputs.push(input);
        self.weights.push(weight);
        self.cumsums.push(new_cumsum);

        // Create R[n] variables
        let pos_n = self.k_max.min(new_cumsum);
        let r_n: Vec<Bool<L>> = (0..pos_n).map(|_| solver.new_var()).collect();
        self.r_vars.push(r_n);

        // Add clauses for the new input only
        self.add_base_clauses(solver, n, n + 1);
    }

    /// Get the assumption literal for enforcing `sum <= k`.
    ///
    /// Returns `!R[n-1][k]` (0-indexed: R[n-1][k] means sum >= k+1,
    /// so !R[n-1][k] means sum <= k).
    ///
    /// Returns None if k >= k_max (constraint is trivially satisfied)
    /// or if there are no inputs.
    pub fn at_most_assumption(&self, k: usize) -> Option<Bool<L>> {
        let n = self.inputs.len();
        if n == 0 {
            return None;
        }
        let last = n - 1;
        if k >= self.r_vars[last].len() {
            return None; // k >= total possible, trivially satisfied
        }
        // sum <= k  <=>  !(sum >= k+1)  <=>  !R[last][k]  (0-indexed)
        Some(!self.r_vars[last][k])
    }

    /// Get the assumption literal for enforcing `sum >= k`.
    ///
    /// Returns `R[n-1][k-1]` (0-indexed).
    pub fn at_least_assumption(&self, k: usize) -> Option<Bool<L>> {
        if k == 0 {
            return Some(true.into());
        }
        let n = self.inputs.len();
        if n == 0 {
            return None;
        }
        let last = n - 1;
        let idx = k - 1; // 0-indexed
        if idx >= self.r_vars[last].len() {
            return None; // k > total possible, impossible
        }
        Some(self.r_vars[last][idx])
    }

    /// Number of inputs currently in the SCPB.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Whether the SCPB has no inputs.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Total weight of all inputs.
    pub fn total_weight(&self) -> usize {
        self.cumsums.last().copied().unwrap_or(0)
    }

    /// Maximum bound this SCPB can handle.
    pub fn k_max(&self) -> usize {
        self.k_max
    }

    /// Evaluate the actual weighted sum from a model assignment.
    pub fn eval_cost(&self, model: &impl Fn(Bool<L>) -> bool) -> usize {
        let mut cost = 0;
        for i in 0..self.inputs.len() {
            if model(self.inputs[i]) {
                cost += self.weights[i];
            }
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile test - actual SAT integration tests would need satcoder test harness
    #[test]
    fn test_scpb_empty() {
        let scpb: SCPB<i32> = SCPB::new_empty(10);
        assert_eq!(scpb.len(), 0);
        assert_eq!(scpb.total_weight(), 0);
        assert!(scpb.is_empty());
    }
}
