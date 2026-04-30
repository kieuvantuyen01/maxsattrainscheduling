use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    time::Instant,
};

#[allow(unused)]
use crate::{
    debug::{ResourceInterval, SolverAction},
    minimize_core,
    problem::Problem,
    trim_core,
};
use satcoder::{
    constraints::{BooleanFormulas, Totalizer},
    prelude::{Binary, SymbolicModel},
    Bool, SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

// -----------------------------------------------------------------------------
// SCL-style encoding for fixed-precedence cliques (rail-path precedence).
//
// In the IAP/DDD formulation, a fixed precedence clique for two consecutive
// visits r \prec q on the SAME train has the staircase form:
//   x_{ir}^p + \sum_{t=1}^{K(p)} x_{iq}^t \le 1.
//
// This solver represents each visit by a monotone ladder of time-point literals
// (Occ::delays): for increasing times t_1 < t_2 < ... we maintain literals
//   d(t)  ==  ("arrival time at least t").
// The ladder clauses ensure d(t_j) => d(t_{j-1}) and d(t_{j+1}) => d(t_j).
//
// Under this representation, the clique above compresses to ONE implication per
// time point (the SCL idea):
//   d_r(t) -> d_q(t + travel_r).
// which forbids all too early choices at q without enumerating them.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct VisitId(u32);

impl From<VisitId> for usize {
    fn from(v: VisitId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for VisitId {
    fn from(x: usize) -> Self {
        VisitId(x as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct ResourceId(u32);

impl From<ResourceId> for usize {
    fn from(v: ResourceId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for ResourceId {
    fn from(x: usize) -> Self {
        ResourceId(x as u32)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum IterationType {
    Objective,
    TravelTimeConflict,
    ResourceConflict,
    TravelAndResourceConflict,
    Solution,
}

#[derive(Default)]
pub struct SolveStats {
    pub n_sat: usize,
    pub n_unsat: usize,
    pub n_travel: usize,
    pub n_conflict: usize,
    pub satsolver: String,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ResourceCliqueRowKey {
    // (visit_id, start, next_incumbent_time). Includes next_incumbent so that
    // when the next-visit's incumbent shifts (changing m_end_lits semantics),
    // the key changes and a fresh tight constraint is added. This avoids the
    // soundness bug where a loose early-iteration constraint would block
    // re-processing the same conflict at later iterations.
    members: Vec<(VisitId, i32, i32)>,
}

#[derive(Clone, Copy, Debug)]
pub struct MaxSatDddLadderSclSettings {
    /// Toggle precedence-graph preprocessing/queue seeding.
    pub use_precedence_graph: bool,
    /// Use SCL-style fixed-precedence rows (d_r(t) -> d_q(t + travel)).
    pub use_scl_fixed_precedence: bool,
    /// Use interval-graph clique-cover conflict encoding (AMO over cliques).
    /// If false, fallback to pairwise conflict generation.
    pub use_interval_graph_conflicts: bool,
    /// Seed fixed precedence rows from earliest time points (only used when SCL is enabled).
    pub seed_scl_from_earliest: bool,
}

impl Default for MaxSatDddLadderSclSettings {
    fn default() -> Self {
        Self {
            use_precedence_graph: true,
            use_scl_fixed_precedence: true,
            use_interval_graph_conflicts: true,
            seed_scl_from_earliest: true,
        }
    }
}

enum Soft<L: satcoder::Lit> {
    Primary,
    Totalizer(Totalizer<L>, usize),
}

fn bits_needed(max_value: usize) -> usize {
    if max_value == 0 {
        1
    } else {
        (usize::BITS as usize) - (max_value.leading_zeros() as usize)
    }
}

fn build_binary_register<L: satcoder::Lit + Copy + 'static>(reg_bits: &[Bool<L>]) -> Binary<L> {
    Binary::from_list(reg_bits.iter().copied())
}

fn build_weighted_binary_term<L: satcoder::Lit + Copy + 'static>(
    lit: Bool<L>,
    weight: usize,
) -> Binary<L> {
    if weight == 0 || lit == false.into() {
        return Binary::constant(0);
    }

    let mut bits = Vec::new();
    let mut remaining = weight;
    while remaining > 0 {
        bits.push(if (remaining & 1usize) == 1usize {
            lit
        } else {
            false.into()
        });
        remaining >>= 1;
    }

    Binary::from_list(bits)
}

fn subtract_binary<L: satcoder::Lit + Copy + 'static>(
    solver: &mut impl SatInstance<L>,
    a: &Binary<L>,
    b: &Binary<L>,
) -> (Binary<L>, Bool<L>) {
    use std::iter::repeat;

    let len = a.clone().into_list().len().max(b.clone().into_list().len());
    let a_bits = a
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    let b_bits = b
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();

    let mut diff_bits = Vec::with_capacity(len);
    let mut borrow = false.into();

    for idx in 0..len {
        let ai = a_bits[idx];
        let bi = b_bits[idx];

        let diff = solver.xor_literal([ai, bi, borrow]);
        let bi_or_borrow = solver.or_literal([bi, borrow]);
        let borrow_from_subtrahend = solver.and_literal([!ai, bi_or_borrow]);
        let borrow_from_borrow = solver.and_literal([bi, borrow]);
        let next_borrow = solver.or_literal([borrow_from_subtrahend, borrow_from_borrow]);

        diff_bits.push(diff);
        borrow = next_borrow;
    }

    (Binary::from_list(diff_bits), borrow)
}

fn binary_le_literal_bits<L: satcoder::Lit + Copy>(
    solver: &mut impl SatInstance<L>,
    a: &[Bool<L>], // MSB -> LSB
    b: &[Bool<L>], // MSB -> LSB
) -> Bool<L> {
    assert_eq!(a.len(), b.len());

    if a.is_empty() {
        return true.into();
    }

    if a.len() == 1 {
        let le_lit = solver.new_var();
        solver.add_clause(vec![!le_lit, !a[0], b[0]]);
        return le_lit;
    }

    let rest = binary_le_literal_bits(solver, &a[1..], &b[1..]);
    let le_lit = solver.new_var();

    solver.add_clause(vec![!le_lit, !a[0], b[0]]);
    solver.add_clause(vec![!le_lit, a[0], b[0], rest]);
    solver.add_clause(vec![!le_lit, !a[0], !b[0], rest]);

    le_lit
}

fn assert_binary_le<L: satcoder::Lit + Copy + 'static>(
    solver: &mut impl SatInstance<L>,
    a: &Binary<L>,
    b: &Binary<L>,
) {
    use std::iter::repeat;

    let len = a.clone().into_list().len().max(b.clone().into_list().len());
    if len == 0 {
        return;
    }

    let mut a_bits = a
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    a_bits.reverse();

    let mut b_bits = b
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    b_bits.reverse();

    let le = binary_le_literal_bits(solver, &a_bits, &b_bits);
    solver.add_clause(vec![le]);
}

struct BinaryObjective<L: satcoder::Lit> {
    reg_bits: Vec<Bool<L>>,
    remaining: Option<Binary<L>>,
}

impl<L: satcoder::Lit + Copy + 'static> BinaryObjective<L> {
    fn new() -> Self {
        Self {
            reg_bits: Vec::new(),
            remaining: None,
        }
    }

    fn add_term(&mut self, solver: &mut impl SatInstance<L>, lit: Bool<L>, weight: usize) {
        if weight == 0 || lit == false.into() {
            return;
        }

        let term = build_weighted_binary_term(lit, weight);
        let current_remaining = self
            .remaining
            .clone()
            .expect("binary objective requires an upper bound before adding terms");
        let (next_remaining, underflow) = subtract_binary(solver, &current_remaining, &term);
        solver.add_clause(vec![!underflow]);
        self.remaining = Some(next_remaining);
    }

    fn ensure_capacity(
        &mut self,
        solver: &mut impl SatInstance<L>,
        soft_constraints: &mut HashMap<Bool<L>, (Soft<L>, usize, usize)>,
        capacity: usize,
    ) {
        let need_bits = bits_needed(capacity);
        let old_len = self.reg_bits.len();

        if need_bits > self.reg_bits.len() {
            while self.reg_bits.len() < need_bits {
                let bit = self.reg_bits.len();
                let reg_bit = solver.new_var();
                let weight = 1usize << bit;
                self.reg_bits.push(reg_bit);
                soft_constraints.insert(!reg_bit, (Soft::Primary, weight, weight));
            }
        }

        if old_len == self.reg_bits.len() {
            if self.remaining.is_none() {
                self.remaining = Some(build_binary_register(&self.reg_bits));
            }
        } else if let Some(remaining) = self.remaining.take() {
            let mut bits = remaining.into_list();
            bits.extend(self.reg_bits[old_len..].iter().copied());
            self.remaining = Some(Binary::from_list(bits));
        } else {
            self.remaining = Some(build_binary_register(&self.reg_bits));
        }
    }
}

#[allow(dead_code)]
fn compute_initial_heuristic_upper_bound<L: satcoder::Lit>(
    mk_env: &impl Fn() -> grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    occupations: &TiVec<VisitId, Occ<L>>,
) -> Result<Option<(i32, Vec<Vec<i32>>)>, SolverError> {
    let initial_solution = extract_solution(problem, occupations);
    let env = mk_env();

    for use_strong_branching in [false, true] {
        if let Some(ub_sol) = heuristic::solve_heuristic_better(
            &env,
            problem,
            delay_cost_type,
            use_strong_branching,
            Some(&initial_solution),
        )? {
            let ub_cost = problem.verify_solution(&ub_sol, delay_cost_type).unwrap();
            return Ok(Some((ub_cost, ub_sol)));
        }
    }

    Ok(None)
}

fn compute_effective_earliest(problem: &Problem) -> Vec<Vec<i32>> {
    let mut effective = Vec::with_capacity(problem.trains.len());

    for train in &problem.trains {
        let mut train_bounds = Vec::with_capacity(train.visits.len());
        let mut propagated_lb: Option<i32> = None;

        for visit in &train.visits {
            let lb = propagated_lb.map_or(visit.earliest, |prev_lb| prev_lb.max(visit.earliest));
            train_bounds.push(lb);
            propagated_lb = Some(lb + visit.travel_time);
        }

        effective.push(train_bounds);
    }

    effective
}

pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        MaxSatDddLadderSclSettings::default(),
        output_stats,
    )
}

pub fn solve_with_settings<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    settings: MaxSatDddLadderSclSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        settings,
        |_| {},
        output_stats,
    )
}

thread_local! { pub static  WATCH : std::cell::RefCell<Option<(usize,usize)>>  = RefCell::new(None);}

use crate::{debug::DebugInfo, problem::DelayCostType, solvers::heuristic};

use crate::solvers::{ddd::costtree::CostTree, SolverError};
pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    debug_out: impl Fn(DebugInfo),
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError>
where
    L: 'static,
{
    solve_debug_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        MaxSatDddLadderSclSettings::default(),
        debug_out,
        output_stats,
    )
}

pub fn solve_debug_with_settings<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    mut solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    settings: MaxSatDddLadderSclSettings,
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    // TODO
    //  - more eager constraint generation
    //    - propagate simple presedences?
    //    - update all conflicts and presedences when adding new time points?
    //    - smt refinement of the simple presedences?
    //  - get rid of the multiple adding of constraints
    //  - cadical doesn't use false polarity, so it can generate unlimited conflicts when cost is maxed. Two trains pushing each other forward.

    let _p = hprof::enter("solver");

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<_>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points = Vec::new();
    let effective_earliest = settings
        .use_precedence_graph
        .then(|| compute_effective_earliest(problem));

    #[allow(unused)]
    let mut core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    #[allow(unused)]
    let mut processed_core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0;
    let mut n_conflict_constraints = 0;

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));
            let earliest = effective_earliest
                .as_ref()
                .map(|bounds| bounds[train_idx][visit_idx])
                .unwrap_or(visit.earliest);

            occupations.push(Occ {
                cost: vec![true.into()],
                cost_tree: CostTree::new(),
                delays: vec![(true.into(), earliest), (false.into(), i32::MAX)],
                incumbent_idx: 0,
            });
            n_timepoints += 1;

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push(visit_id);
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), earliest));

            for t in problem.trains[train_idx]
                .visit_cost_threshold_times(delay_cost_type, visit_idx, earliest)
            {
                let (var, is_new) = occupations[visit_id].time_point(&mut solver, t);
                if is_new {
                    n_timepoints += 1;
                    new_time_points.push((visit_id, var, t));
                }
            }
        }
    }

    // The first iteration (0) does not need a solve call; we
    // know it's SAT because there are no constraints yet.
    let mut iteration = 1;
    let mut is_sat = true;

    let mut total_cost = 0;
    let mut soft_constraints = HashMap::new();
    let mut debug_actions = Vec::new();
    // let mut cost_var_names: HashMap<Bool<L>, String> = HashMap::new();

    // Rows already added for interval-graph conflict encoding.
    let mut added_resource_clique_rows: HashSet<ResourceCliqueRowKey> = HashSet::new();
    // Rows already added for fixed-precedence encoding: (visit_id, time).
    let mut fixed_prec_rows: HashSet<(VisitId, i32)> = HashSet::new();
    // let mut priorities: Vec<(VisitId, VisitId)> = Vec::new();

    // Optional: seed fixed-precedence (travel-time) constraints from the earliest
    // time points to reduce the number of "travel-time conflict" iterations.
    if settings.seed_scl_from_earliest {
        for visit_id in visits.keys() {
            let (train_idx, visit_idx) = visits[visit_id];
            if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
                continue;
            }
            let (in_var, in_t) = occupations[visit_id].delays[0];
            if settings.use_precedence_graph {
                propagate_precedence(
                    &mut solver,
                    problem,
                    &visits,
                    &mut occupations,
                    &mut new_time_points,
                    &mut fixed_prec_rows,
                    visit_id,
                    in_var,
                    in_t,
                    settings.use_scl_fixed_precedence,
                );
            } else if settings.use_scl_fixed_precedence {
                let _ = add_fixed_precedence_row(
                    &mut solver,
                    problem,
                    &visits,
                    &mut occupations,
                    &mut new_time_points,
                    &mut fixed_prec_rows,
                    visit_id,
                    in_var,
                    in_t,
                    settings.use_scl_fixed_precedence,
                );
            }
        }
    }

    // Async heuristic: instead of blocking on initial heuristic computation
    // at init (5-30s of Gurobi call time), spawn the heuristic thread
    // immediately and let SAT solver start working in parallel. The heuristic
    // result flows in via `heur_thread`'s channel during the main loop, where
    // we update `best_heur` and inject solution timepoints when it arrives.
    const USE_HEURISTIC: bool = true;
    let mut best_heur: Option<(i32, Vec<Vec<i32>>)> = None;
    let mut injected_heuristic_cost: Option<i32> = None;

    let heur_thread = USE_HEURISTIC.then(|| {
        let (sol_in_tx, sol_in_rx) = std::sync::mpsc::channel();
        let (sol_out_tx, sol_out_rx) = std::sync::mpsc::channel();
        let problem = problem.clone();
        heuristic::spawn_heuristic_thread(mk_env, sol_in_rx, problem, delay_cost_type, sol_out_tx);
        (sol_in_tx, sol_out_rx)
    });

    loop {
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_heur.map(|(c, _)| c).unwrap_or(i32::MAX);
            println!("TIMEOUT LB={} UB={}", total_cost, ub);

            do_output_stats(
                &mut output_stats,
                iteration,
                &iteration_types,
                &stats,
                &occupations,
                start_time,
                solver_time,
                total_cost,
                ub,
            );
            return Err(SolverError::Timeout);
        }

        let _p = hprof::enter("iteration");
        if is_sat {
            // println!("Iteration {} conflict detection starting...", iteration);

            if let Some((sol_tx, sol_rx)) = heur_thread.as_ref() {
                let sol = extract_solution(problem, &occupations);
                let _ = sol_tx.send(sol);

                while let Ok((ub_cost, ub_sol)) = sol_rx.try_recv() {
                    if ub_cost < total_cost as i32 {
                        println!(
                            "HEURISTIC UB={} is below current LB={}; keeping UB but skipping LB==UB termination this iteration",
                            ub_cost, total_cost
                        );
                        if ub_cost < best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                            best_heur = Some((ub_cost, ub_sol));
                        }
                        continue;
                    }
                    if ub_cost == total_cost as i32 {
                        println!("HEURISTIC UB=LB");
                        println!("TERMINATE HEURISTIC");
                        println!(
                            "MAXSAT ITERATIONS {}  {}",
                            n_conflict_constraints, iteration
                        );
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            total_cost,
                            ub_cost,
                        );

                        return Ok((ub_sol, stats));
                    }

                    if ub_cost < best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                        best_heur = Some((ub_cost, ub_sol));
                    }
                }

                // NOTE: We intentionally do NOT inject heuristic solution
                // timepoints here. Empirically, `inject_solution_timepoints_maxsat`
                // causes formula bloat for the `Continuous` objective (each
                // injected timepoint → cost var → CostTree growth → larger
                // formula → slower SAT calls → more timeouts). Ladder
                // (`maxsat_ddd_ladder`) also receives heuristic UB updates but
                // does NOT inject, and outperforms ladder_scl on cont benchmarks
                // because of this. We match that behavior: use the heuristic
                // purely for UB tracking and the UB=LB termination check above.
                // The fallback injection at core.len()==0 (later in the loop)
                // is kept as a last-resort for edge cases.
            }

            let mut found_travel_time_conflict = false;
            let mut found_resource_conflict = false;

            // let mut touched_intervals = visits.keys().collect::<Vec<_>>();

            for visit_id in touched_intervals.iter().copied() {
                let _p = hprof::enter("travel time check");
                let (train_idx, visit_idx) = visits[visit_id];
                let next_visit: Option<VisitId> =
                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                        Some((usize::from(visit_id) + 1).into())
                    } else {
                        None
                    };

                // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
                let t1_in = occupations[visit_id].incumbent_time();
                let visit = problem.trains[train_idx].visits[visit_idx];

                if let Some(next_visit) = next_visit {
                    let v1 = &occupations[visit_id];
                    let v2 = &occupations[next_visit];
                    let t1_out = v2.incumbent_time();

                    // TRAVEL TIME CONFLICT
                    if t1_in + visit.travel_time > t1_out {
                        found_travel_time_conflict = true;
                        // println!(
                        //     "  - TRAVEL time conflict train{} visit{} resource{} in{} travel{} out{}",
                        //     train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                        // );

                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));

                        // Insert/update precedence row for this time point.
                        let in_var = v1.delays[v1.incumbent_idx].0;
                        let in_t = v1.incumbent_time();
                        let _ = add_fixed_precedence_row(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            visit_id,
                            in_var,
                            in_t,
                            settings.use_scl_fixed_precedence,
                        );
                        stats.n_travel += 1;
                    }

                    // let v1 = &occupations[visit];
                    // let v2 = &occupations[next_visit];

                    // // TRAVEL TIME CONFLICT
                    // if t1_in + travel_time < t1_out {
                    //     found_conflict = true;
                    //     println!(
                    //                             "  - TRAVEL OVERtime conflict train{} visit{} resource{} in{} travel{} out{}",
                    //                             train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                    //                         );

                    //     // Insert the new time point.
                    //     let t1_in_var = v1.delays[v1.incumbent].0;
                    //     let new_t = v1.incumbent_time() + travel_time;
                    //     let (t1_earliest_out_var, t1_is_new) =
                    //         occupations[next_visit].time_point(&mut solver, new_t);

                    //     // T1_IN delay implies T1_EARLIEST_OUT delay.
                    //     SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);
                    //     // The new timepoint might have a cost.
                    //     if t1_is_new {
                    //         new_time_points.push((next_visit, t1_in_var, new_t));
                    //     }
                    // }
                }
            }

            // println!("Solving conflicts in iteration {}", iteration);

            // SOLVE ALL SIMPLE PRESEDENCES BEFORE CONFLICTS
            // if !found_conflict {

            let _p = hprof::enter("conflict check");
            if settings.use_interval_graph_conflicts {
                #[derive(Clone, Copy)]
                struct ActiveInterval {
                    visit_id: VisitId,
                    train_idx: usize,
                    start: i32,
                    end: i32,
                }

                let touched_set: HashSet<VisitId> = touched_intervals.iter().copied().collect();
                let mut touched_positions: HashMap<VisitId, Vec<usize>> = HashMap::new();
                for (idx, visit_id) in touched_intervals.iter().copied().enumerate() {
                    touched_positions.entry(visit_id).or_default().push(idx);
                }

                // Only process resource pairs impacted by touched visits.
                let mut relevant_resource_pairs: HashSet<(usize, usize)> = HashSet::new();
                for &visit_id in &touched_intervals {
                    let (train_idx, visit_idx) = visits[visit_id];
                    let resource = problem.trains[train_idx].visits[visit_idx].resource_id;
                    if let Some(conflicting_resources) = conflicts.get(&resource) {
                        for &other in conflicting_resources {
                            if resource <= other {
                                relevant_resource_pairs.insert((resource, other));
                            } else {
                                relevant_resource_pairs.insert((other, resource));
                            }
                        }
                    }
                }

                let mut retain_touched = vec![false; touched_intervals.len()];
                // Per-iteration cache: (visit_id, tau+1) → active literal.
                // Reused across cliques in the same iteration to share aux vars.
                let mut active_lit_cache: HashMap<(VisitId, i32), Bool<L>> = HashMap::new();

                // Collect all clique candidates across resource pairs first, then
                // process them in severity order with a per-iteration budget.
                let mut clique_candidates: Vec<(i64, Vec<ActiveInterval>, i32, usize, usize)> =
                    Vec::new();

                for (resource_a, resource_b) in relevant_resource_pairs.into_iter() {
                    // Early filter: skip pairs where either resource has no visits.
                    // Saves O(V) scan + allocation for empty resources.
                    if resource_a >= resource_visits.len()
                        || resource_b >= resource_visits.len()
                        || resource_visits[resource_a].is_empty()
                        || resource_visits[resource_b].is_empty()
                    {
                        continue;
                    }

                    let mut group_intervals = Vec::new();
                    for &resource in [resource_a, resource_b].iter() {
                        if resource >= resource_visits.len() {
                            continue;
                        }
                        if resource == resource_b
                            && resource_a == resource_b
                            && !group_intervals.is_empty()
                        {
                            continue;
                        }
                        for &visit_id in &resource_visits[resource] {
                            let (train_idx, visit_idx) = visits[visit_id];
                            let start = occupations[visit_id].incumbent_time();
                            // The occupation interval of a visit on a resource is
                            // [start, start + travel_time).  Do NOT use the next
                            // visit's incumbent_time() here — that value includes
                            // slack accumulated during DDD iterations and inflates
                            // the interval, producing wrong clique membership.
                            let end =
                                start + problem.trains[train_idx].visits[visit_idx].travel_time;
                            if end <= start {
                                continue;
                            }
                            group_intervals.push(ActiveInterval {
                                visit_id,
                                train_idx,
                                start,
                                end,
                            });
                        }
                    }

                    if group_intervals.len() < 2 {
                        continue;
                    }

                    let mut taus: Vec<i32> = group_intervals.iter().map(|it| it.start).collect();
                    taus.sort_unstable();
                    taus.dedup();

                    for tau in taus {
                        let members: Vec<ActiveInterval> = group_intervals
                            .iter()
                            .copied()
                            .filter(|it| it.start <= tau && tau < it.end)
                            .collect();

                        if members.len() <= 1 {
                            continue;
                        }
                        if !members.iter().any(|m| touched_set.contains(&m.visit_id)) {
                            continue;
                        }

                        let mut trains_in_clique: HashSet<usize> = HashSet::new();
                        for m in &members {
                            trains_in_clique.insert(m.train_idx);
                        }
                        if trains_in_clique.len() <= 1 {
                            continue;
                        }

                        // Severity = member_count * overlap_length. Prioritize large
                        // cliques with long overlaps (most binding conflicts).
                        let overlap_min_end = members.iter().map(|m| m.end).min().unwrap();
                        let overlap_length = (overlap_min_end - tau).max(1) as i64;
                        let severity = (members.len() as i64) * overlap_length;

                        clique_candidates.push((
                            severity,
                            members,
                            tau,
                            resource_a,
                            resource_b,
                        ));
                    }
                }

                // Sort by severity descending — process most "dangerous" cliques
                // first. Do NOT filter: low-severity cliques are still real
                // resource conflicts that must be resolved. Skipping them could
                // prevent convergence or cause false-optimal returns for
                // configurations where small overlaps matter for feasibility.
                clique_candidates.sort_by(|a, b| b.0.cmp(&a.0));

                // Per-iteration budget to prevent combinatorial blowup when many
                // cliques are detected at once. Remaining cliques will be re-detected
                // and processed in subsequent iterations (smart dedup via
                // `added_resource_clique_rows` prevents double-adding).
                const MAX_CLIQUES_PER_ITER: usize = 100;
                let mut cliques_processed = 0usize;

                for (_severity, members, _tau, _resource_a, _resource_b) in clique_candidates {
                    if cliques_processed >= MAX_CLIQUES_PER_ITER {
                        found_resource_conflict = true;
                        break;
                    }

                    let mut member_key: Vec<(VisitId, i32, i32)> = members
                        .iter()
                        .map(|m| {
                            let (train_idx, visit_idx) = visits[m.visit_id];
                            let next_incumbent =
                                if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                    let next_id: VisitId =
                                        (usize::from(m.visit_id) + 1).into();
                                    occupations[next_id].incumbent_time()
                                } else {
                                    i32::MAX
                                };
                            (m.visit_id, m.start, next_incumbent)
                        })
                        .collect();
                    member_key.sort_unstable_by_key(|(v, t, _)| (v.0, *t));
                    let row_key = ResourceCliqueRowKey {
                        members: member_key,
                    };
                    if !added_resource_clique_rows.insert(row_key) {
                        continue;
                    }

                    found_resource_conflict = true;
                    stats.n_conflict += 1;

                    for m in &members {
                        if let Some(idxs) = touched_positions.get(&m.visit_id) {
                            for &idx in idxs {
                                retain_touched[idx] = true;
                            }
                        }
                    }

                    // Fast-path for 2-member cliques (the common case in train
                    // scheduling): emit a direct pairwise monotone clause
                    // instead of Tseitin-encoded `active_i(tau)` aux vars + AMO.
                    // Saves 6 Tseitin clauses + 2 aux vars per clique.
                    if members.len() == 2 {
                        let mi = members[0];
                        let mj = members[1];

                        // Capture m_end literals BEFORE any timepoint creation.
                        let m_end_lit_of = |visit_id: VisitId,
                                            occs: &TiVec<VisitId, Occ<L>>|
                         -> Bool<L> {
                            let (train_idx, visit_idx) = visits[visit_id];
                            if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                let next_id: VisitId =
                                    (usize::from(visit_id) + 1).into();
                                occs[next_id].delays[occs[next_id].incumbent_idx].0
                            } else {
                                true.into()
                            }
                        };

                        let m_end_i = m_end_lit_of(mi.visit_id, &occupations);
                        let m_end_j = m_end_lit_of(mj.visit_id, &occupations);

                        let delay_i = get_delay_lit_at(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            settings.use_scl_fixed_precedence,
                            mi.visit_id,
                            mj.end,
                        );
                        let delay_j = get_delay_lit_at(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            settings.use_scl_fixed_precedence,
                            mj.visit_id,
                            mi.end,
                        );

                        solver.add_clause(vec![!m_end_i, !m_end_j, delay_i, delay_j]);
                    } else {
                        // Sequential AMO via Tseitin-encoded "active_i(tau)" aux vars.
                        // Choose tau+1 = min(member.end) so the AMO forces at least
                        // all but one member to start at or after this time. Uses
                        // monotone delay literals capturing BOTH start and end.
                        let tau_plus_1 = members.iter().map(|m| m.end).min().unwrap();

                        // Build lits and split by train in a single pass for BIS encoding.
                        // The clique is bipartite when exactly 2 trains are involved:
                        //   side A = intervals of train_a, side B = intervals of train_b.
                        // For 3+ trains we collect all lits and fall back to hybrid AMO.
                        let mut seen_trains: Vec<usize> = Vec::new();
                        for m in &members {
                            if !seen_trains.contains(&m.train_idx) {
                                seen_trains.push(m.train_idx);
                            }
                        }

                        if seen_trains.len() == 2 {
                            let train_a = seen_trains[0];
                            let mut side_a: Vec<Bool<L>> = Vec::new();
                            let mut side_b: Vec<Bool<L>> = Vec::new();

                            for m in &members {
                                let lit = build_active_lit(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    &mut active_lit_cache,
                                    settings.use_scl_fixed_precedence,
                                    m.visit_id,
                                    tau_plus_1,
                                );
                                if m.train_idx == train_a {
                                    side_a.push(lit);
                                } else {
                                    side_b.push(lit);
                                }
                            }
                            // BIS(K_{a,b}): |side_a|+|side_b| clauses instead of O(k log k).
                            add_bipartite_amo(&mut solver, &side_a, &side_b);
                        } else {
                            // 3+ trains sharing resource: collect all lits, use hybrid AMO.
                            // (Correct but uses more clauses; rare in practice.)
                            let mut all_lits: Vec<Bool<L>> = Vec::with_capacity(members.len());
                            for m in &members {
                                let lit = build_active_lit(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    &mut active_lit_cache,
                                    settings.use_scl_fixed_precedence,
                                    m.visit_id,
                                    tau_plus_1,
                                );
                                all_lits.push(lit);
                            }
                            add_hybrid_amo(&mut solver, &all_lits);
                        }
                    }
                    n_conflict_constraints += 1;
                    cliques_processed += 1;
                }

                let mut new_touched = Vec::new();
                for (idx, visit_id) in touched_intervals.into_iter().enumerate() {
                    if retain_touched[idx] {
                        new_touched.push(visit_id);
                    }
                }
                touched_intervals = new_touched;
            } else {
                let mut deconflicted_train_pairs: HashSet<(usize, usize)> = HashSet::new();
                touched_intervals.retain(|visit_id| {
                    let visit_id = *visit_id;
                    let (train_idx, visit_idx) = visits[visit_id];
                    let next_visit: Option<VisitId> =
                        if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                            Some((usize::from(visit_id) + 1).into())
                        } else {
                            None
                        };

                    let t1_in = occupations[visit_id].incumbent_time();
                    let visit = problem.trains[train_idx].visits[visit_idx];
                    let mut retain = false;

                    if let Some(conflicting_resources) = conflicts.get(&visit.resource_id) {
                        for other_resource in conflicting_resources.iter().copied() {
                            let t1_out = next_visit
                                .map(|nx| occupations[nx].incumbent_time())
                                .unwrap_or(t1_in + visit.travel_time);

                            for other_visit in resource_visits[other_resource].iter().copied() {
                                if usize::from(visit_id) == usize::from(other_visit) {
                                    continue;
                                }

                                let v2 = &occupations[other_visit];
                                let t2_in = v2.incumbent_time();
                                let (other_train_idx, other_visit_idx) = visits[other_visit];

                                if other_train_idx == train_idx {
                                    continue;
                                }

                                let other_next_visit: Option<VisitId> = if other_visit_idx + 1
                                    < problem.trains[other_train_idx].visits.len()
                                {
                                    Some((usize::from(other_visit) + 1).into())
                                } else {
                                    None
                                };

                                let t2_out = other_next_visit
                                    .map(|v| occupations[v].incumbent_time())
                                    .unwrap_or_else(|| {
                                        let other_v =
                                            problem.trains[other_train_idx].visits[other_visit_idx];
                                        t2_in + other_v.travel_time
                                    });

                                if t1_out <= t2_in || t2_out <= t1_in {
                                    continue;
                                }

                                if !deconflicted_train_pairs.insert((train_idx, other_train_idx))
                                    || !deconflicted_train_pairs
                                        .insert((other_train_idx, train_idx))
                                {
                                    retain = true;
                                    continue;
                                }

                                found_resource_conflict = true;
                                stats.n_conflict += 1;

                                let (delay_t2, t2_is_new) =
                                    occupations[other_visit].time_point(&mut solver, t1_out);
                                let (delay_t1, t1_is_new) =
                                    occupations[visit_id].time_point(&mut solver, t2_out);

                                if t1_is_new {
                                    new_time_points.push((visit_id, delay_t1, t2_out));
                                }
                                let _ = add_fixed_precedence_row(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    visit_id,
                                    delay_t1,
                                    t2_out,
                                    settings.use_scl_fixed_precedence,
                                );

                                if t2_is_new {
                                    new_time_points.push((other_visit, delay_t2, t1_out));
                                }
                                let _ = add_fixed_precedence_row(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    other_visit,
                                    delay_t2,
                                    t1_out,
                                    settings.use_scl_fixed_precedence,
                                );

                                let t1_out_lit = next_visit
                                    .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                    .unwrap_or_else(|| true.into());
                                let t2_out_lit = other_next_visit
                                    .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                    .unwrap_or_else(|| true.into());

                                n_conflict_constraints += 1;
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!t1_out_lit, !t2_out_lit, delay_t1, delay_t2],
                                );
                                retain = true;
                            }
                        }
                    }
                    retain
                });
            }

            // touched_intervals.clear();
            // assert!(touched_intervals.is_empty());
            // }

            let iterationtype = if found_travel_time_conflict && found_resource_conflict {
                IterationType::TravelAndResourceConflict
            } else if found_travel_time_conflict {
                IterationType::TravelTimeConflict
            } else if found_resource_conflict {
                // println!("Iteration {}", iteration);
                IterationType::ResourceConflict
            } else {
                IterationType::Solution
            };

            *iteration_types.entry(iterationtype).or_default() += 1;

            if !(found_resource_conflict || found_travel_time_conflict) {
                // Incumbent times are feasible and optimal.

                const USE_LP_MINIMIZE: bool = false;

                let trains = if !USE_LP_MINIMIZE {
                    extract_solution(problem, &occupations)
                } else {
                    // let p = priorities
                    //     .into_iter()
                    //     .map(|(a, b)| (visits[a], visits[b]))
                    //     .collect();
                    // minimize::minimize_solution(env, problem, p)?
                    panic!()
                };

                println!(
                    "Finished with cost {} iterations {} solver {:?}",
                    total_cost, iteration, solver
                );
                println!("Core size bins {:?}", core_sizes);
                println!("Iteration types {:?}", iteration_types);
                debug_out(DebugInfo {
                    iteration,
                    actions: std::mem::take(&mut debug_actions),
                    solution: extract_solution(problem, &occupations),
                });

                stats.satsolver = format!("{:?}", solver);

                println!(
                    "STATS {} {} {} {} {} {} {} {}",
                    /* iter */ iteration,
                    /* objective iters */
                    iteration_types.get(&IterationType::Objective).unwrap_or(&0),
                    /* travel iters */
                    iteration_types
                        .get(&IterationType::TravelTimeConflict)
                        .unwrap_or(&0),
                    /* resource iters */
                    iteration_types
                        .get(&IterationType::ResourceConflict)
                        .unwrap_or(&0),
                    /* both iters */
                    iteration_types
                        .get(&IterationType::TravelAndResourceConflict)
                        .unwrap_or(&0),
                    /* solution iters */
                    iteration_types.get(&IterationType::Solution).unwrap_or(&0),
                    /* num traveltime */ stats.n_travel,
                    /* num conflicts */ stats.n_conflict,
                );

                do_output_stats(
                    &mut output_stats,
                    iteration,
                    &iteration_types,
                    &stats,
                    &occupations,
                    start_time,
                    solver_time,
                    total_cost,
                    total_cost,
                );

                println!("VARSCLAUSES {:?}", solver);

                println!(
                    "MAXSAT ITERATIONS {}  {}",
                    n_conflict_constraints, iteration
                );
                return Ok((trains, stats));
            }
        }
        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];
            // let resource = problem.trains[train_idx].visits[visit_idx].resource_id;

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {
                // println!(
                //     "new var for t{} v{} t{} cost{}",
                //     train_idx, visit_idx, new_t, new_timepoint_cost
                // );

                // let var_name = format!(
                //     "t{}v{}t{}cost{}",
                //     train_idx, visit_idx, new_t, new_timepoint_cost
                // );

                const USE_COST_TREE: bool = true;
                if !USE_COST_TREE {
                    for cost in occupations[visit].cost.len()..=new_timepoint_cost {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = SatInstance::new_var(&mut solver);

                        SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);

                        occupations[visit].cost.push(next_cost_var);
                        assert!(cost + 1 == occupations[visit].cost.len());

                        soft_constraints.insert(!next_cost_var, (Soft::Primary, 1, 1));
                        // println!(
                        //     "Extending t{}v{} to cost {} {:?}",
                        //     train_idx, visit_idx, cost, next_cost_var
                        // );
                    }

                    SatInstance::add_clause(
                        &mut solver,
                        vec![
                            !new_timepoint_var,
                            occupations[visit].cost[new_timepoint_cost],
                        ],
                    );

                    // println!("  highest cost {}", occupations[visit].cost.len() - 1);
                } else {
                    // if let Some((weight, cost_var)) = occupations[visit].cost_tree.add_cost(
                    //     &mut solver,
                    //     new_timepoint_var,
                    //     new_timepoint_cost,
                    // ) {
                    //     assert!(weight > 0);
                    //     soft_constraints.insert(!cost_var, (Soft::Delay, weight, weight));
                    // }

                    // Direct insertion in callback — no Vec buffering overhead.
                    // Matches the ladder (non-SCL) pattern for lower allocation cost.
                    occupations[visit].cost_tree.add_cost(
                        &mut solver,
                        new_timepoint_var,
                        new_timepoint_cost,
                        &mut |weight, cost_var| {
                            soft_constraints
                                .insert(!cost_var, (Soft::Primary, weight, weight));
                        },
                    );
                }
            }

            // set the cost for this new time point.

            // WATCH.with(|x| {
            //     if *x.borrow() == Some((train_idx, visit_idx)) {
            // println!(
            //     "Soft constraint for t{}-v{}-r{} t{} cost{} lit{:?}",
            //     train_idx, visit_idx, resource, new_t, new_var_cost, new_var
            // );
            // println!(
            //     "   new var implies cost {}=>{:?}",
            //     new_var_cost, occupations[visit].cost[new_var_cost]
            // );
            //     }
            // });
            // println!(
            //     "Soft constraint for t{}-v{}-r{} t{} cost{} lit{:?}",
            //     train_idx, visit_idx, resource, new_t, new_var_cost, new_var
            // );
            // println!(
            //     "   new var implies cost {}=>{:?}",
            //     new_var_cost, occupations[visit].cost[new_var_cost]
            // );
            // SatInstance::add_clause(
            //     &mut solver,
            //     vec![!new_var, occupations[visit].cost[new_var_cost]],
            // );
        }

        let mut n_assumps = soft_constraints.len();
        let mut assumptions = soft_constraints
            .iter()
            .map(|(k, (_, w, _))| (*k, *w))
            .collect::<Vec<_>>();
        assumptions.sort_by(|a, b| b.1.cmp(&a.1));

        log::info!(
            "solving it{} with {} timepoints {} conflicts",
            iteration,
            n_timepoints,
            n_conflict_constraints
        );
        let core = loop {
            let solve_start = Instant::now();
            let result = {
                let _p = hprof::enter("sat check");
                SatSolverWithCore::solve_with_assumptions(
                    &mut solver,
                    assumptions.iter().map(|(k, _)| *k).take(n_assumps),
                )
            };
            solver_time += solve_start.elapsed();

            // println!("solving done");
            match result {
                satcoder::SatResultWithCore::Sat(_) if n_assumps < soft_constraints.len() => {
                    n_assumps += 20;
                }
                satcoder::SatResultWithCore::Sat(model) => {
                    is_sat = true;
                    stats.n_sat += 1;
                    let _p = hprof::enter("update times");

                    for (visit, this_occ) in occupations.iter_mut_enumerated() {
                        // let old_time = this_occ.incumbent_time();
                        let mut touched = false;

                        while model.value(&this_occ.delays[this_occ.incumbent_idx + 1].0) {
                            this_occ.incumbent_idx += 1;
                            touched = true;
                        }
                        while !model.value(&this_occ.delays[this_occ.incumbent_idx].0) {
                            this_occ.incumbent_idx -= 1;
                            touched = true;
                        }
                        let (_train_idx, visit_idx) = visits[visit];

                        // let resource = problem.trains[train_idx].visits[visit_idx].resource_id;
                        // let new_time = this_occ.incumbent_time();

                        // WATCH.with(|x| {
                        //     if *x.borrow() == Some((train_idx, visit_idx))  && touched {
                        //         println!("Delays {:?}", this_occ.delays);
                        //         println!(
                        //             "Updated t{}-v{}-r{}  t={}-->{}",
                        //             train_idx, visit_idx, resource, old_time, new_time
                        //         );
                        //     }
                        // });

                        if touched {
                            // println!(
                            //     "Updated t{}-v{}-r{}  t={}-->{}",
                            //     train_idx, visit_idx, resource, old_time, new_time
                            // );

                            // We are really interested not in the visits, but the resource occupation
                            // intervals. Therefore, also the previous visit has been touched by this visit.
                            if visit_idx > 0 {
                                let prev_visit = (Into::<usize>::into(visit) - 1).into();
                                if touched_intervals.last() != Some(&prev_visit) {
                                    touched_intervals.push(prev_visit);
                                }
                            }
                            touched_intervals.push(visit);
                        }

                        // let cost = this_occ
                        //     .cost
                        //     .iter()
                        //     .map(|c| if model.value(c) { 1 } else { 0 })
                        //     .sum::<isize>()
                        //     - 1;
                        // if cost > 0 {
                        //     // println!("t{}-v{}  cost={}", train_idx, visit_idx, cost);
                        // }
                    }

                    const USE_LOCAL_MINIMIZE: bool = true;
                    if USE_LOCAL_MINIMIZE {
                        let mut last_mod = 0;
                        let mut i = 0;
                        let occs_len = occupations.len();
                        assert!(visits.len() == occupations.len());
                        while last_mod < occs_len {
                            let mut touched = false;
                            // println!("i = {} (l={})",i, visits.len());

                            let visit_id = VisitId(i % occs_len as u32);
                            while occupations[visit_id].incumbent_idx > 0 {
                                // We can always leave earlier, so the critical interval is
                                // from this event to the next.

                                let t1_in = occupations[visit_id].delays
                                    [occupations[visit_id].incumbent_idx]
                                    .1;
                                let t1_in_new = occupations[visit_id].delays
                                    [occupations[visit_id].incumbent_idx - 1]
                                    .1;

                                let (train_idx, visit_idx) = visits[visit_id];
                                let visit = problem.trains[train_idx].visits[visit_idx];

                                let next_visit: Option<VisitId> =
                                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                        Some((usize::from(visit_id) + 1).into())
                                    } else {
                                        None
                                    };

                                let prev_visit: Option<VisitId> = if visit_idx > 0 {
                                    Some((usize::from(visit_id) - 1).into())
                                } else {
                                    None
                                };

                                let t1_prev_earliest_out = prev_visit
                                    .map(|v| {
                                        let (tidx, vidx) = visits[v];
                                        let travel_time =
                                            problem.trains[tidx].visits[vidx].travel_time;
                                        occupations[v].incumbent_time() + travel_time
                                    })
                                    .unwrap_or(i32::MIN);

                                let travel_ok = t1_prev_earliest_out <= t1_in_new;

                                let t1_out = next_visit
                                    .map(|nx| occupations[nx].incumbent_time())
                                    .unwrap_or(t1_in + visit.travel_time);

                                let can_reduce = travel_ok
                                    && conflicts
                                        .get(&visit.resource_id)
                                        .iter()
                                        .flat_map(|rs| rs.iter())
                                        .copied()
                                        .all(|other_resource| {
                                            resource_visits[other_resource]
                                                .iter()
                                                .copied()
                                                .filter(|other_visit| {
                                                    usize::from(visit_id)
                                                        != usize::from(*other_visit)
                                                })
                                                .filter(|other_visit| {
                                                    visits[*other_visit].0 != train_idx
                                                })
                                                .all(|other_visit| {
                                                    let v2 = &occupations[other_visit];
                                                    let t2_in = v2.incumbent_time();
                                                    let (other_train_idx, other_visit_idx) =
                                                        visits[other_visit];
                                                    let other_next_visit: Option<VisitId> =
                                                        if other_visit_idx + 1
                                                            < problem.trains[other_train_idx]
                                                                .visits
                                                                .len()
                                                        {
                                                            Some(
                                                                (usize::from(other_visit) + 1)
                                                                    .into(),
                                                            )
                                                        } else {
                                                            None
                                                        };

                                                    let t2_out = other_next_visit
                                                        .map(|v| occupations[v].incumbent_time())
                                                        .unwrap_or_else(|| {
                                                            let other_visit = problem.trains
                                                                [other_train_idx]
                                                                .visits[other_visit_idx];
                                                            t2_in + other_visit.travel_time
                                                        });
                                                    t1_out <= t2_in || t2_out <= t1_in_new
                                                })
                                        });

                                if can_reduce {
                                    // println!("REDUCE {} {} {}", train_idx, visit_idx, occupations[visit_id].incumbent_time());
                                    occupations[visit_id].incumbent_idx -= 1;
                                    touched = true;
                                    last_mod = 0;
                                } else {
                                    break;
                                }
                            }

                            i += 1;

                            if touched {
                                let visit_idx = visits[visit_id].1;
                                if visit_idx > 0 {
                                    let prev_visit = (Into::<usize>::into(visit_id) - 1).into();
                                    if touched_intervals.last() != Some(&prev_visit) {
                                        touched_intervals.push(prev_visit);
                                    }
                                }
                                touched_intervals.push(visit_id);
                            } else {
                                last_mod += 1;
                            }
                        }
                    }

                    // println!(
                    //     "Touched {}/{} occupations",
                    //     touched_intervals.len(),
                    //     occupations.len()
                    // );

                    // priorities = conflict_vars
                    //     .iter()
                    //     .filter_map(|(pair, l)| {
                    //         let has_choice = model.value(l);
                    //         let has_time = occupations[pair.0].incumbent_time()
                    //             < occupations[pair.1].incumbent_time();
                    //         (has_choice && has_time).then(|| *pair)
                    //     })
                    //     .collect::<Vec<_>>();
                    // println!("Pri {:?}", priorities);

                    debug_out(DebugInfo {
                        iteration,
                        actions: std::mem::take(&mut debug_actions),
                        solution: extract_solution(problem, &occupations),
                    });

                    break None;
                }
                satcoder::SatResultWithCore::Unsat(core) => {
                    is_sat = false;
                    stats.n_unsat += 1;
                    break Some(core);
                }
            }
        };

        if let Some(core) = core {
            let _p = hprof::enter("treat core");
            // println!("Got core length {}", core.len());
            // Do weighted RC2

            if core.len() == 0 {
                if !settings.use_precedence_graph {
                    if let Some((ub_cost, ub_sol)) = best_heur.as_ref() {
                        if injected_heuristic_cost != Some(*ub_cost) {
                            inject_solution_timepoints_maxsat(
                                &mut solver,
                                problem,
                                &visits,
                                &mut occupations,
                                &mut new_time_points,
                                &mut fixed_prec_rows,
                                settings.use_scl_fixed_precedence,
                                ub_sol,
                            );
                            injected_heuristic_cost = Some(*ub_cost);
                            iteration += 1;
                            continue;
                        }
                    }
                }
                return Err(SolverError::NoSolution); // UNSAT
            }

            let core = core.iter().map(|c| Bool::Lit(*c)).collect::<Vec<_>>();

            // println!("Core size {}", core.len());
            // // *core_sizes.entry(core.len()).or_default() += 1;
            // trim_core(&mut core, &mut solver);
            // minimize_core(&mut core, &mut solver);
            // println!("Post core size {}", core.len());

            // *processed_core_sizes.entry(core.len()).or_default() += 1;
            // println!("  pre sizes {:?}", core_sizes);
            // println!("  post sizes {:?}", processed_core_sizes);
            *iteration_types.entry(IterationType::Objective).or_default() += 1;
            debug_actions.push(SolverAction::Core(core.len()));

            let min_weight = core.iter().map(|c| soft_constraints[c].1).min().unwrap();
            // let max_weight = core.iter().map(|c| soft_constraints[c].1).max().unwrap();
            assert!(min_weight >= 1);

            // println!("Core sz{} weight range {} -- {} assumps {}/{}",  core.len(), min_weight, max_weight, n_assumps, soft_constraints.len());

            for c in core.iter() {
                let (soft, cost, original_cost) = soft_constraints.remove(c).unwrap();

                // let soft_str = match &soft {
                //     Soft::Delay => "delay".to_string(),
                //     Soft::Totalizer(_, b) => format!("totalizer w/bound={}", b),
                // };

                // println!("  * {:?} {:?} {} {}", c, cost_var_names.get(c), soft_str, cost);

                assert!(cost >= min_weight);
                let new_cost = cost - min_weight;
                // assert!(new_cost >= 0);
                // assert!(original_cost == 1);
                match soft {
                    Soft::Primary => {
                        if new_cost > 0 {
                            // println!("  ** Reducing delay cost from {} to {}", cost, new_cost);
                            soft_constraints.insert(*c, (Soft::Primary, new_cost, original_cost));
                        } else {
                            // println!("  ** Removing delay cost {}", cost);
                        }
                        /* primary soft constraint, when we relax to new_cost=0 we are done */
                    }
                    Soft::Totalizer(mut tot, bound) => {
                        // panic!();
                        if new_cost > 0 {
                            // println!("  ** Reducing totalizer cost from {} to {}", cost, new_cost);

                            soft_constraints
                                .insert(*c, (Soft::Totalizer(tot, bound), new_cost, original_cost));
                        } else {
                            // panic!();
                            // totalizer: need to extend its bound
                            let new_bound = bound + 1;
                            // println!("Increasing totalizer bound to {}", new_bound);
                            tot.increase_bound(&mut solver, new_bound as u32);
                            if new_bound < tot.rhs().len() {
                                // println!(
                                //     "  ** Expanding totalizer original cost {}",
                                //     original_cost
                                // );

                                // let mut name = cost_var_names[c].clone();
                                // name.push_str(&format!("<={}", new_bound));
                                // cost_var_names.insert(!tot.rhs()[new_bound], name);

                                soft_constraints.insert(
                                    !tot.rhs()[new_bound], // tot <= 2, 3, 4...
                                    (
                                        Soft::Totalizer(tot, new_bound),
                                        original_cost,
                                        original_cost,
                                    ),
                                );
                            } else {
                                // println!("  ** Totalizer fully expanded {}", cost);
                            }
                        }
                    }
                }
            }

            // println!(
            //     "increasing cost from {} to {}",
            //     total_cost,
            //     total_cost + min_weight
            // );
            total_cost += min_weight as i32;
            println!("    LB={}", total_cost);

            if total_cost as i32 == best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                println!("TERMINATE HEURISTIC");
                println!(
                    "MAXSAT ITERATIONS {}  {}",
                    n_conflict_constraints, iteration
                );
                do_output_stats(
                    &mut output_stats,
                    iteration,
                    &iteration_types,
                    &stats,
                    &occupations,
                    start_time,
                    solver_time,
                    total_cost,
                    total_cost,
                );

                return Ok((best_heur.unwrap().1, stats));
            }

            if core.len() > 1 {
                let bound = 1;
                let tot = Totalizer::count(&mut solver, core.iter().map(|c| !*c), bound as u32);
                assert!(bound < tot.rhs().len());

                // let mut name = String::new();
                // for c in core {
                //     name.push_str(&format!("{}+", cost_var_names[&c]));
                // }
                // name.push_str(&format!("<={}", bound));
                // cost_var_names.insert(!tot.rhs()[bound], name);
                soft_constraints.insert(
                    !tot.rhs()[bound], // tot <= 1
                    (Soft::Totalizer(tot, bound), min_weight, min_weight),
                );
            } else {
                // panic!();
                SatInstance::add_clause(&mut solver, vec![!core[0]]);
            }
        }

        iteration += 1;
        // println!("iteration {}", iteration);
    }
}

fn do_output_stats<L: satcoder::Lit>(
    output_stats: &mut impl FnMut(String, serde_json::Value),
    iteration: usize,
    iteration_types: &BTreeMap<IterationType, usize>,
    stats: &SolveStats,
    occupations: &TiVec<VisitId, Occ<L>>,
    start_time: Instant,
    solver_time: std::time::Duration,
    lb: i32,
    ub: i32,
) {
    output_stats("iterations".to_string(), iteration.into());
    output_stats(
        "objective_iters".to_string(),
        (*iteration_types.get(&IterationType::Objective).unwrap_or(&0)).into(),
    );
    output_stats(
        "travel_iters".to_string(),
        (*iteration_types
            .get(&IterationType::TravelTimeConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats(
        "resource_iters".to_string(),
        (*iteration_types
            .get(&IterationType::ResourceConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats(
        "travel_and_resource_iters".to_string(),
        (*iteration_types
            .get(&IterationType::TravelAndResourceConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats("num_traveltime".to_string(), stats.n_travel.into());
    output_stats("num_conflicts".to_string(), stats.n_travel.into());
    output_stats(
        "num_time_points".to_string(),
        occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .sum::<usize>()
            .into(),
    );
    output_stats(
        "max_time_points".to_string(),
        occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .max()
            .unwrap()
            .into(),
    );
    output_stats(
        "avg_time_points".to_string(),
        ((occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .sum::<usize>() as f64)
            / (occupations.len() as f64))
            .into(),
    );

    output_stats(
        "total_time".to_string(),
        start_time.elapsed().as_secs_f64().into(),
    );
    output_stats("solver_time".to_string(), solver_time.as_secs_f64().into());
    output_stats(
        "algorithm_time".to_string(),
        (start_time.elapsed().as_secs_f64() - solver_time.as_secs_f64()).into(),
    );
    output_stats("lb".to_string(), lb.into());
    output_stats("ub".to_string(), ub.into());
}

fn extract_solution<L: satcoder::Lit>(
    problem: &Problem,
    occupations: &TiVec<VisitId, Occ<L>>,
) -> Vec<Vec<i32>> {
    let _p = hprof::enter("extract solution");
    let mut trains = Vec::new();
    let mut i = 0;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_times = Vec::new();
        for _ in train.visits.iter().enumerate() {
            train_times.push(occupations[VisitId(i)].incumbent_time());
            i += 1;
        }

        let visit = problem.trains[train_idx].visits[train_times.len() - 1];
        let last_t = train_times[train_times.len() - 1] + visit.travel_time;
        train_times.push(last_t);

        trains.push(train_times);
    }
    trains
}

fn inject_solution_timepoints_maxsat<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    use_scl_fixed_precedence: bool,
    sol: &[Vec<i32>],
) {
    let mut flat_visit = 0usize;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            let visit_id = VisitId(flat_visit as u32);
            flat_visit += 1;
            let time = sol[train_idx][visit_idx];
            let (lit, is_new) = occupations[visit_id].time_point(solver, time);
            if is_new {
                new_time_points.push((visit_id, lit, time));
            }
            if visit_idx + 1 < train.visits.len() {
                let _ = add_fixed_precedence_row(
                    solver,
                    problem,
                    visits,
                    occupations,
                    new_time_points,
                    fixed_prec_rows,
                    visit_id,
                    lit,
                    time,
                    use_scl_fixed_precedence,
                );
            }
        }
    }
}

#[derive(Debug)]
struct Occ<L: satcoder::Lit> {
    cost: Vec<Bool<L>>,
    cost_tree: CostTree<L>,
    delays: Vec<(Bool<L>, i32)>,
    incumbent_idx: usize,
}

impl<L: satcoder::Lit> Occ<L> {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent_idx].1
    }

    pub fn time_point(&mut self, solver: &mut impl SatInstance<L>, t: i32) -> (Bool<L>, bool) {
        // The inserted time should be between the neighboring times.
        // assert!(idx == 0 || self.delays[idx - 1].1 < t);
        // assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        // println!("idx {} t {}   delays{:?}", idx, t, self.delays);

        assert!(idx > 0 || t == self.delays[0].1); // cannot insert before the earliest time.
        assert!(idx < self.delays.len()); // cannot insert after infinity.

        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t || (idx > 0 && self.delays[idx - 1].1 == t) {
            return (self.delays[idx].0, false);
        }

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        // Keep `incumbent_idx` pointing to the same logical timepoint after
        // insertion. If we inserted at or before the incumbent's array slot,
        // the old incumbent shifted forward by one — compensate so callers
        // reading `incumbent_time()` or `delays[incumbent_idx]` still see
        // the same data they had before this call. Without this, an insertion
        // before the incumbent leaves `delays[incumbent_idx]` pointing to the
        // newly-inserted (wrong) timepoint, causing travel-time violations.
        if idx <= self.incumbent_idx {
            self.incumbent_idx += 1;
        }

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}


fn add_sequential_amo<L: satcoder::Lit>(solver: &mut impl SatInstance<L>, lits: &[Bool<L>]) {
    match lits.len() {
        0 | 1 => return,
        2 => {
            solver.add_clause(vec![!lits[0], !lits[1]]);
            return;
        }
        _ => {}
    }

    let mut prefix = Vec::with_capacity(lits.len() - 1);
    for _ in 0..(lits.len() - 1) {
        prefix.push(solver.new_var());
    }

    solver.add_clause(vec![!lits[0], prefix[0]]);
    for i in 1..(lits.len() - 1) {
        solver.add_clause(vec![!lits[i], prefix[i]]);
        solver.add_clause(vec![!prefix[i - 1], prefix[i]]);
        solver.add_clause(vec![!lits[i], !prefix[i - 1]]);
    }
    solver.add_clause(vec![!lits[lits.len() - 1], !prefix[prefix.len() - 1]]);
}

fn add_pairwise_amo<L: satcoder::Lit>(solver: &mut impl SatInstance<L>, lits: &[Bool<L>]) {
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            solver.add_clause(vec![!lits[i], !lits[j]]);
        }
    }
}

fn add_hybrid_amo<L: satcoder::Lit>(solver: &mut impl SatInstance<L>, lits: &[Bool<L>]) {
    const PAIRWISE_AMO_MAX_SIZE: usize = 5;
    if lits.len() <= PAIRWISE_AMO_MAX_SIZE {
        add_pairwise_amo(solver, lits);
    } else {
        add_sequential_amo(solver, lits);
    }
}

<<<<<<< HEAD
/// Monotone delay literal `visit_id.start ≥ t`.
/// Returns `true.into()` if t ≤ earliest (always satisfied),
/// `false.into()` if t ≥ infinity sentinel (never satisfied),
/// otherwise creates the timepoint (if new) and returns its literal.
fn get_delay_lit_at<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    use_scl_fixed_precedence: bool,
    visit_id: VisitId,
    t: i32,
) -> Bool<L> {
    let (earliest_t, last_t) = {
        let occ = &occupations[visit_id];
        (occ.delays[0].1, occ.delays[occ.delays.len() - 1].1)
    };
    if t <= earliest_t {
        return true.into();
    }
    if t >= last_t {
        return false.into();
    }
    let (lit, is_new) = occupations[visit_id].time_point(solver, t);
    if is_new {
        new_time_points.push((visit_id, lit, t));
        let _ = add_fixed_precedence_row(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            visit_id,
            lit,
            t,
            use_scl_fixed_precedence,
        );
    }
    lit
}

/// Build a sound "active at tau" aux variable via Tseitin:
///   active_i = (start_i ≤ tau) ∧ (end_i > tau)
///           = !delay_i(tau+1) ∧ delay_next_i(tau+1)
/// For the last visit of a train (no next visit), end_i = start_i + travel,
/// so `end_i > tau` ⟺ `start_i ≥ tau - travel + 1`, encoded as `delay_i(tau+1-travel)`.
///
/// Cached per (visit_id, tau+1) to avoid duplicate aux vars.
fn build_active_lit<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    active_cache: &mut HashMap<(VisitId, i32), Bool<L>>,
    use_scl_fixed_precedence: bool,
    visit_id: VisitId,
    tau_plus_1: i32,
) -> Bool<L> {
    if let Some(&lit) = active_cache.get(&(visit_id, tau_plus_1)) {
        return lit;
    }

    let (train_idx, visit_idx) = visits[visit_id];

    let delay_start = get_delay_lit_at(
        solver,
        problem,
        visits,
        occupations,
        new_time_points,
        fixed_prec_rows,
        use_scl_fixed_precedence,
        visit_id,
        tau_plus_1,
    );

    let delay_end = if visit_idx + 1 < problem.trains[train_idx].visits.len() {
        let next_id: VisitId = (usize::from(visit_id) + 1).into();
        get_delay_lit_at(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            use_scl_fixed_precedence,
            next_id,
            tau_plus_1,
        )
    } else {
        let travel = problem.trains[train_idx].visits[visit_idx].travel_time;
        get_delay_lit_at(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            use_scl_fixed_precedence,
            visit_id,
            tau_plus_1 - travel,
        )
    };

    let active = solver.new_var();
    // active → !delay_start
    solver.add_clause(vec![!active, !delay_start]);
    // active → delay_end
    solver.add_clause(vec![!active, delay_end]);
    // (!delay_start ∧ delay_end) → active
    solver.add_clause(vec![active, delay_start, !delay_end]);

    active_cache.insert((visit_id, tau_plus_1), active);
    active
=======
/// BIS(K_{a,b}) encoding — Subercaseaux (2025), Equation 5.
///
/// For a bipartite clique with sides A and B:
///   introduce one aux variable y, then add:
///     ∀ l ∈ A : ¬l ∨ y          (any A selected ⇒ y)
///     ∀ l ∈ B : ¬y ∨ ¬l          (y ⇒ no B selected)
///
/// Uses |A|+|B| clauses.  Falls back to pairwise when one side is a
/// singleton (no auxiliary variable needed).
fn add_bipartite_amo<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    side_a: &[Bool<L>],
    side_b: &[Bool<L>],
) {
    match (side_a.len(), side_b.len()) {
        (0, _) | (_, 0) => { /* nothing to constrain */ }
        (1, 1) => {
            solver.add_clause(vec![!side_a[0], !side_b[0]]);
        }
        (1, _) => {
            for &lb in side_b {
                solver.add_clause(vec![!side_a[0], !lb]);
            }
        }
        (_, 1) => {
            for &la in side_a {
                solver.add_clause(vec![!la, !side_b[0]]);
            }
        }
        _ => {
            // General BIS: introduce y = “some A-literal is true”.
            let y = solver.new_var();
            for &la in side_a {
                solver.add_clause(vec![!la, y]);   // ¬la ∨ y
            }
            for &lb in side_b {
                solver.add_clause(vec![!y, !lb]);  // ¬y ∨ ¬lb
            }
        }
    }
>>>>>>> 4d74b905 (update bis encoding)
}

/// Add a fixed precedence row for one chosen time point and return the
/// propagated successor time point for further queue-based propagation.
fn add_fixed_precedence_row<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    added: &mut HashSet<(VisitId, i32)>,
    visit_id: VisitId,
    in_var: Bool<L>,
    in_t: i32,
    use_scl_fixed_precedence: bool,
) -> Option<(VisitId, Bool<L>, i32)> {
    if !added.insert((visit_id, in_t)) {
        return None;
    }

    let (train_idx, visit_idx) = visits[visit_id];
    if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
        return None;
    }

    let travel = problem.trains[train_idx].visits[visit_idx].travel_time;
    let next_visit: VisitId = (usize::from(visit_id) + 1).into();
    let req_t = in_t + travel;

    let earliest_next = occupations[next_visit].delays[0].1;
    if req_t <= earliest_next {
        return Some((next_visit, true.into(), earliest_next));
    }

    let (req_var, is_new) = occupations[next_visit].time_point(solver, req_t);
    if use_scl_fixed_precedence {
        // Hybrid SCL (long-chain SCL variant): for short chains (≤ threshold)
        // use the single Plain implication — monotonicity propagation handles
        // it cheaply with no formula growth. For long chains (> threshold)
        // expand into 3-literal chain clauses through the delay ladder so
        // unit propagation reaches the far end in one shot rather than
        // multi-hop through monotonicity.
        const SCL_LONG_CHAIN_THRESHOLD: usize = 5;
        let idx = occupations[next_visit]
            .delays
            .partition_point(|(_, t0)| *t0 < req_t);
        if idx > SCL_LONG_CHAIN_THRESHOLD {
            for i in 0..idx {
                let lit_i = occupations[next_visit].delays[i].0;
                let lit_next = occupations[next_visit].delays[i + 1].0;
                solver.add_clause(vec![!in_var, !lit_i, lit_next]);
            }
        } else {
            solver.add_clause(vec![!in_var, req_var]);
        }
    } else {
        // Plain encoding: a single 2-literal implication. Relies on the
        // monotonicity chain among delay literals (already established in
        // `Occ::time_point`) for correct forward propagation.
        solver.add_clause(vec![!in_var, req_var]);
    }
    if is_new {
        new_time_points.push((next_visit, req_var, req_t));
    }

    Some((next_visit, req_var, req_t))
}

fn propagate_precedence<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    added: &mut HashSet<(VisitId, i32)>,
    start_visit: VisitId,
    start_var: Bool<L>,
    start_t: i32,
    use_scl_fixed_precedence: bool,
) {
    let mut queue = VecDeque::from([(start_visit, start_var, start_t)]);

    while let Some((visit_id, in_var, in_t)) = queue.pop_front() {
        if let Some(next) = add_fixed_precedence_row(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            added,
            visit_id,
            in_var,
            in_t,
            use_scl_fixed_precedence,
        ) {
            queue.push_back(next);
        }
    }
}
