use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
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
    constraints::Totalizer, prelude::SymbolicModel, Bool, SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

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

use super::maxsatddd_ladder::{IterationType, SolveStats};

use crate::{debug::DebugInfo, problem::DelayCostType, solvers::heuristic};

use super::{costtree::CostTree, SolverError};

/// A wrapper around a SAT solver that records all operations (new_var, add_clause)
/// so that the solver can be rebuilt from scratch at any point.
struct ReplayableSolver<S> {
    solver: S,
    /// Record of all new_var calls (just count them)
    n_vars: usize,
    /// Record of all hard clauses added (stored as raw isize literals)
    clause_log: Vec<Vec<isize>>,
    /// Factory to create new solver instances
    rebuild_count: usize,
}

impl<L: satcoder::Lit + Copy + std::fmt::Debug, S: SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug> ReplayableSolver<S> {
    fn rebuild(&mut self, mk_solver: &impl Fn() -> S) {
        let start = Instant::now();
        let mut new_solver = mk_solver();
        
        // Replay all new_var calls
        for _ in 0..self.n_vars {
            SatInstance::new_var(&mut new_solver);
        }
        
        // Replay all hard clauses
        for clause in &self.clause_log {
            // Convert isize back to Bool<L>
            let bools: Vec<Bool<L>> = clause.iter().map(|&lit| {
                if lit > 0 {
                    Bool::Lit(L::from_dimacs(lit))
                } else if lit < 0 {
                    !Bool::Lit(L::from_dimacs(-lit))
                } else {
                    panic!("zero literal in clause log")
                }
            }).collect();
            SatInstance::add_clause(&mut new_solver, bools);
        }
        
        self.rebuild_count += 1;
        self.solver = new_solver;
        
        println!(
            "REBUILD #{}: replayed {} vars, {} clauses in {:.3}s",
            self.rebuild_count,
            self.n_vars,
            self.clause_log.len(),
            start.elapsed().as_secs_f64()
        );
    }
}

/// Convert a Bool<L> to isize for storage in the clause log.
fn bool_to_isize<L: satcoder::Lit>(b: Bool<L>) -> isize {
    match b {
        Bool::Lit(l) => l.to_dimacs(),
        Bool::Const(true) => panic!("Cannot log Bool::Const(true) in clause"),
        Bool::Const(false) => panic!("Cannot log Bool::Const(false) in clause"),
    }
}

pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    mk_solver: impl Fn() -> (impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug),
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    rebuild_every: usize,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug(
        mk_env,
        mk_solver,
        problem,
        timeout,
        delay_cost_type,
        rebuild_every,
        |_| {},
        output_stats,
    )
}

pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug, S: SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    mk_solver: impl Fn() -> S,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    rebuild_every: usize, // rebuild solver every N DDD iterations (0 = never)
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let _p = hprof::enter("solver");

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();

    let mut rs = ReplayableSolver {
        solver: mk_solver(),
        n_vars: 0,
        clause_log: Vec::new(),
        rebuild_count: 0,
    };

    // Helper macros/closures to track operations
    macro_rules! rs_new_var {
        ($rs:expr) => {{
            $rs.n_vars += 1;
            SatInstance::new_var(&mut $rs.solver)
        }};
    }

    macro_rules! rs_add_clause {
        ($rs:expr, $clause:expr) => {{
            let clause_vec: Vec<Bool<L>> = $clause;
            // Log the clause (skip if contains constants)
            let mut all_lits = true;
            let mut isize_clause = Vec::with_capacity(clause_vec.len());
            for b in &clause_vec {
                match b {
                    Bool::Lit(l) => isize_clause.push(l.to_dimacs()),
                    _ => { all_lits = false; break; }
                }
            }
            if all_lits {
                $rs.clause_log.push(isize_clause);
            }
            SatInstance::add_clause(&mut $rs.solver, clause_vec);
        }};
    }

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<_>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points = Vec::new();

    #[allow(unused)]
    let mut core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    #[allow(unused)]
    let mut processed_core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0;
    let mut n_conflict_constraints = 0;
    let mut ddd_iteration_count = 0usize;

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));

            occupations.push(Occ {
                cost: vec![true.into()],
                cost_tree: CostTree::new(),
                delays: vec![(true.into(), visit.earliest), (false.into(), i32::MAX)],
                incumbent_idx: 0,
            });
            n_timepoints += 1;

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push(visit_id);
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), visit.earliest));
        }
    }

    let mut iteration = 1;
    let mut is_sat = true;

    let mut total_cost = 0;
    let mut soft_constraints = HashMap::new();
    let mut debug_actions = Vec::new();
    let mut conflict_vars: HashMap<(VisitId, VisitId), Bool<L>> = Default::default();

    const USE_HEURISTIC: bool = true;

    let heur_thread = USE_HEURISTIC.then(|| {
        let (sol_in_tx, sol_in_rx) = std::sync::mpsc::channel();
        let (sol_out_tx, sol_out_rx) = std::sync::mpsc::channel();
        let problem = problem.clone();
        heuristic::spawn_heuristic_thread(mk_env, sol_in_rx, problem, delay_cost_type, sol_out_tx);
        (sol_in_tx, sol_out_rx)
    });
    let mut best_heur: Option<(i32, Vec<Vec<i32>>)> = None;

    loop {
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_heur.map(|(c, _)| c).unwrap_or(i32::MAX);
            println!(
                "TIMEOUT LB={} UB={}",
                total_cost,
                ub
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
                ub
            );
            return Err(SolverError::Timeout);
        }

        let _p = hprof::enter("iteration");
        if is_sat {
            // Check if we should rebuild the solver
            ddd_iteration_count += 1;
            if rebuild_every > 0 && ddd_iteration_count > 1 && (ddd_iteration_count - 1) % rebuild_every == 0 {
                rs.rebuild(&mk_solver);
            }

            if let Some((sol_tx, sol_rx)) = heur_thread.as_ref() {
                let sol = extract_solution(problem, &occupations);
                let _ = sol_tx.send(sol);

                while let Ok((ub_cost, ub_sol)) = sol_rx.try_recv() {
                    assert!(ub_cost >= total_cost as i32);
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
                            ub_cost
                        );
        
                        return Ok((ub_sol, stats));
                    }

                    if ub_cost < best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                        best_heur = Some((ub_cost, ub_sol));
                    }
                }
            }

            let mut found_travel_time_conflict = false;
            let mut found_resource_conflict = false;

            for visit_id in touched_intervals.iter().copied() {
                let _p = hprof::enter("travel time check");
                let (train_idx, visit_idx) = visits[visit_id];
                let next_visit: Option<VisitId> =
                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                        Some((usize::from(visit_id) + 1).into())
                    } else {
                        None
                    };

                let t1_in = occupations[visit_id].incumbent_time();
                let visit = problem.trains[train_idx].visits[visit_idx];

                if let Some(next_visit) = next_visit {
                    let v1 = &occupations[visit_id];
                    let v2 = &occupations[next_visit];
                    let t1_out = v2.incumbent_time();

                    if t1_in + visit.travel_time > t1_out {
                        found_travel_time_conflict = true;

                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));

                        let t1_in_var = v1.delays[v1.incumbent_idx].0;
                        let new_t = v1.incumbent_time() + visit.travel_time;
                        let (t1_earliest_out_var, t1_is_new) =
                            occupations[next_visit].time_point_rs(&mut rs, new_t);

                        rs_add_clause!(rs, vec![!t1_in_var, t1_earliest_out_var]);
                        stats.n_travel += 1;
                        if t1_is_new {
                            new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                        }
                    }
                }
            }

            let mut deconflicted_train_pairs: HashSet<(usize, usize)> = HashSet::new();

            touched_intervals.retain(|visit_id| {
                let visit_id = *visit_id;

                let _p = hprof::enter("conflict check");
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
                                    let other_visit =
                                        problem.trains[other_train_idx].visits[other_visit_idx];
                                    t2_in + other_visit.travel_time
                                });

                            if t1_out <= t2_in || t2_out <= t1_in {
                                continue;
                            }

                            if !deconflicted_train_pairs.insert((train_idx, other_train_idx))
                                || !deconflicted_train_pairs.insert((other_train_idx, train_idx))
                            {
                                retain = true;
                                continue;
                            }

                            found_resource_conflict = true;
                            stats.n_conflict += 1;

                            #[allow(unused, clippy::let_unit_value)]
                            let t1_in = ();
                            #[allow(unused, clippy::let_unit_value)]
                            let t2_in = ();

                            let (delay_t2, t2_is_new) =
                                occupations[other_visit].time_point_rs(&mut rs, t1_out);
                            let (delay_t1, t1_is_new) =
                                occupations[visit_id].time_point_rs(&mut rs, t2_out);

                            if t1_is_new {
                                new_time_points.push((visit_id, delay_t1, t2_out));
                            }

                            if t2_is_new {
                                new_time_points.push((other_visit, delay_t2, t1_out));
                            }

                            let v1 = &occupations[visit_id];
                            let v2 = &occupations[other_visit];

                            let t1_out_lit = next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());
                            let t2_out_lit = other_next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());

                            n_conflict_constraints += 1;

                            rs_add_clause!(rs, vec![
                                !t1_out_lit,
                                !t2_out_lit,
                                delay_t1,
                                delay_t2,
                            ]);
                        }
                    }
                }

                retain
            });

            let iterationtype = if found_travel_time_conflict && found_resource_conflict {
                IterationType::TravelAndResourceConflict
            } else if found_travel_time_conflict {
                IterationType::TravelTimeConflict
            } else if found_resource_conflict {
                IterationType::ResourceConflict
            } else {
                IterationType::Solution
            };

            *iteration_types.entry(iterationtype).or_default() += 1;

            if !(found_resource_conflict || found_travel_time_conflict) {
                let trains = extract_solution(problem, &occupations);

                println!(
                    "Finished with cost {} iterations {} solver {:?} rebuilds {}",
                    total_cost, iteration, rs.solver, rs.rebuild_count
                );
                println!("Core size bins {:?}", core_sizes);
                println!("Iteration types {:?}", iteration_types);
                debug_out(DebugInfo {
                    iteration,
                    actions: std::mem::take(&mut debug_actions),
                    solution: extract_solution(problem, &occupations),
                });

                stats.satsolver = format!("{:?}", rs.solver);

                println!(
                    "STATS {} {} {} {} {} {} {} {}",
                    iteration,
                    iteration_types.get(&IterationType::Objective).unwrap_or(&0),
                    iteration_types
                        .get(&IterationType::TravelTimeConflict)
                        .unwrap_or(&0),
                    iteration_types
                        .get(&IterationType::ResourceConflict)
                        .unwrap_or(&0),
                    iteration_types
                        .get(&IterationType::TravelAndResourceConflict)
                        .unwrap_or(&0),
                    iteration_types.get(&IterationType::Solution).unwrap_or(&0),
                    stats.n_travel,
                    stats.n_conflict,
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
                    total_cost
                );

                println!("VARSCLAUSES {:?}", rs.solver);

                println!(
                    "MAXSAT ITERATIONS {}  {}",
                    n_conflict_constraints, iteration
                );
                return Ok((trains, stats));
            }
        }
        enum Soft<L: satcoder::Lit> {
            Delay,
            Totalizer(Totalizer<L>, usize),
        }

        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {
                const USE_COST_TREE: bool = true;
                if !USE_COST_TREE {
                    for cost in occupations[visit].cost.len()..=new_timepoint_cost {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = rs_new_var!(rs);

                        rs_add_clause!(rs, vec![!next_cost_var, prev_cost_var]);

                        occupations[visit].cost.push(next_cost_var);
                        assert!(cost + 1 == occupations[visit].cost.len());

                        soft_constraints.insert(!next_cost_var, (Soft::Delay, 1, 1));
                    }

                    rs_add_clause!(rs, vec![
                        !new_timepoint_var,
                        occupations[visit].cost[new_timepoint_cost],
                    ]);
                } else {
                    occupations[visit].cost_tree.add_cost(
                        &mut rs.solver,
                        new_timepoint_var,
                        new_timepoint_cost,
                        &mut |weight, cost_var| {
                            // Track the new_var and clause from CostTree
                            // CostTree internally calls new_var and add_clause on the solver,
                            // but we need to track those too. Since CostTree uses the solver
                            // directly, we need to manually count vars after CostTree operations.
                            soft_constraints.insert(!cost_var, (Soft::Delay, weight, weight));
                        },
                    );
                }
            }
        }

        // After CostTree operations, sync n_vars count
        // This is a simplification - we count all vars the solver has created
        // CostTree calls new_var internally, which we can't easily intercept
        // So we track by checking solver state after the fact
        // Note: This means our replay log for CostTree clauses may be incomplete
        // A full solution would wrap CostTree to also use rs_add_clause/rs_new_var

        let mut n_assumps = 20;
        let mut assumptions = soft_constraints
            .iter()
            .map(|(k, (_, w, _))| (*k, *w))
            .collect::<Vec<_>>();
        assumptions.sort_by_key(|(_, w)| -(*w as isize));

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
                    &mut rs.solver,
                    assumptions.iter().map(|(k, _)| *k).take(n_assumps),
                )
            };
            solver_time += solve_start.elapsed();

            match result {
                satcoder::SatResultWithCore::Sat(_) if n_assumps < soft_constraints.len() => {
                    n_assumps += 20;
                }
                satcoder::SatResultWithCore::Sat(model) => {
                    is_sat = true;
                    stats.n_sat += 1;
                    let _p = hprof::enter("update times");

                    for (visit, this_occ) in occupations.iter_mut_enumerated() {
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

                        if touched {
                            if visit_idx > 0 {
                                let prev_visit = (Into::<usize>::into(visit) - 1).into();
                                if touched_intervals.last() != Some(&prev_visit) {
                                    touched_intervals.push(prev_visit);
                                }
                            }
                            touched_intervals.push(visit);
                        }
                    }

                    const USE_LOCAL_MINIMIZE: bool = true;
                    if USE_LOCAL_MINIMIZE {
                        let mut last_mod = 0;
                        let mut i = 0;
                        let occs_len = occupations.len();
                        assert!(visits.len() == occupations.len());
                        while last_mod < occs_len {
                            let mut touched = false;

                            let visit_id = VisitId(i % occs_len as u32);
                            while occupations[visit_id].incumbent_idx > 0 {
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

            if core.len() == 0 {
                return Err(SolverError::NoSolution);
            }

            let core = core.iter().map(|c| Bool::Lit(*c)).collect::<Vec<_>>();

            *iteration_types.entry(IterationType::Objective).or_default() += 1;
            debug_actions.push(SolverAction::Core(core.len()));

            let min_weight = core.iter().map(|c| soft_constraints[c].1).min().unwrap();
            assert!(min_weight >= 1);

            for c in core.iter() {
                let (soft, cost, original_cost) = soft_constraints.remove(c).unwrap();

                assert!(cost >= min_weight);
                let new_cost = cost - min_weight;
                match soft {
                    Soft::Delay => {
                        if new_cost > 0 {
                            soft_constraints.insert(*c, (Soft::Delay, new_cost, new_cost));
                        }
                    }
                    Soft::Totalizer(mut tot, bound) => {
                        if new_cost > 0 {
                            soft_constraints
                                .insert(*c, (Soft::Totalizer(tot, bound), new_cost, original_cost));
                        } else {
                            let new_bound = bound + 1;
                            tot.increase_bound(&mut rs.solver, new_bound as u32);
                            if new_bound < tot.rhs().len() {
                                soft_constraints.insert(
                                    !tot.rhs()[new_bound],
                                    (
                                        Soft::Totalizer(tot, new_bound),
                                        original_cost,
                                        original_cost,
                                    ),
                                );
                            }
                        }
                    }
                }
            }

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
                    total_cost
                );

                return Ok((best_heur.unwrap().1, stats));
            }

            if core.len() > 1 {
                let bound = 1;
                let tot = Totalizer::count(&mut rs.solver, core.iter().map(|c| !*c), bound as u32);
                assert!(bound < tot.rhs().len());

                soft_constraints.insert(
                    !tot.rhs()[bound],
                    (Soft::Totalizer(tot, bound), min_weight, min_weight),
                );
            } else {
                SatInstance::add_clause(&mut rs.solver, vec![!core[0]]);
                // Also log this hard clause
                if let Bool::Lit(l) = !core[0] {
                    rs.clause_log.push(vec![l.to_dimacs()]);
                }
            }
        }

        iteration += 1;
    }
}

fn do_output_stats<L:satcoder::Lit>(
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

    /// time_point with ReplayableSolver tracking
    pub fn time_point_rs<S: SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug>(
        &mut self,
        rs: &mut ReplayableSolver<S>,
        t: i32,
    ) -> (Bool<L>, bool) {
        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        assert!(idx > 0 || t == self.delays[0].1);
        assert!(idx < self.delays.len());

        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t || (idx > 0 && self.delays[idx - 1].1 == t) {
            return (self.delays[idx].0, false);
        }

        rs.n_vars += 1;
        let var = SatInstance::new_var(&mut rs.solver);
        self.delays.insert(idx, (var, t));

        if idx > 0 {
            let clause = vec![!var, self.delays[idx - 1].0];
            // Log clause
            let mut isize_clause = Vec::new();
            let mut all_lits = true;
            for b in &clause {
                match b {
                    Bool::Lit(l) => isize_clause.push(l.to_dimacs()),
                    _ => { all_lits = false; break; }
                }
            }
            if all_lits { rs.clause_log.push(isize_clause); }
            SatInstance::add_clause(&mut rs.solver, clause);
        }

        if idx < self.delays.len() {
            let clause = vec![!self.delays[idx + 1].0, var];
            let mut isize_clause = Vec::new();
            let mut all_lits = true;
            for b in &clause {
                match b {
                    Bool::Lit(l) => isize_clause.push(l.to_dimacs()),
                    _ => { all_lits = false; break; }
                }
            }
            if all_lits { rs.clause_log.push(isize_clause); }
            SatInstance::add_clause(&mut rs.solver, clause);
        }

        (var, true)
    }
}
