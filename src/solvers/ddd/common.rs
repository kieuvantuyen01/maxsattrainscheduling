// src/solvers/ddd/common.rs
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};
use satcoder::{Bool, SatInstance};
use typed_index_collections::TiVec;

use crate::{
    debug::{ResourceInterval, SolverAction},
    problem::{DelayCostType, Problem},
};
use super::costtree::CostTree;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VisitId(pub u32);
impl From<VisitId> for usize { fn from(v: VisitId) -> Self { v.0 as usize } }
impl From<usize> for VisitId { fn from(x: usize) -> Self { VisitId(x as u32) } }

#[derive(Clone, Copy, Debug)]
pub struct ResourceId(pub u32);
impl From<ResourceId> for usize { fn from(v: ResourceId) -> Self { v.0 as usize } }
impl From<usize> for ResourceId { fn from(x: usize) -> Self { ResourceId(x as u32) } }

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

#[derive(Debug)]
pub struct Occ<L: satcoder::Lit> {
    pub cost: Vec<Bool<L>>,
    pub cost_tree: CostTree<L>,
    pub delays: Vec<(Bool<L>, i32)>,
    pub incumbent_idx: usize,
}

impl<L: satcoder::Lit> Occ<L> {
    pub fn incumbent_time(&self) -> i32 { self.delays[self.incumbent_idx].1 }

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

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}

pub struct CostTerm<L: satcoder::Lit> {
    pub var: Bool<L>,     // TRUE => pay weight
    pub weight: usize,
}

#[derive(Clone, Copy)]
pub enum CostMode { Ladder, CostTree }

pub struct DddState<L: satcoder::Lit> {
    pub visits: TiVec<VisitId, (usize, usize)>,
    pub train_visit_ids: Vec<Vec<VisitId>>,
    pub resource_visits: Vec<Vec<VisitId>>,
    pub occupations: TiVec<VisitId, Occ<L>>,
    pub touched_intervals: Vec<VisitId>,
    pub conflicts: HashMap<usize, Vec<usize>>,
    pub new_time_points: Vec<(VisitId, Bool<L>, i32)>,
    pub conflict_vars: HashMap<(VisitId, VisitId), Bool<L>>,
    pub n_timepoints: usize,
    pub n_conflict_constraints: usize,
}

impl<L: satcoder::Lit + Copy> DddState<L> {
    pub fn new(problem: &Problem, solver: &mut impl SatInstance<L>) -> Self {
        // === COPY đoạn init từ solve_debug vào đây ===
        // - build conflicts map
        // - loop trains/visits: push VisitId, occupations, resource_visits, touched_intervals, new_time_points
        unimplemented!()
    }

    pub fn extract_solution(&self, problem: &Problem) -> Vec<Vec<i32>> {
        // === COPY nguyên extract_solution (đổi occupations -> self.occupations) ===
        unimplemented!()
    }

    pub fn inject_solution_timepoints(
        &mut self,
        solver: &mut impl SatInstance<L>,
        sol: &[Vec<i32>],
    ) {
        // (Khuyến nghị cho SAT-budget) thêm timepoints từ UB heuristic vào encoding
        // - dùng self.train_visit_ids[train][visit] để lấy VisitId
        // - gọi occ.time_point(solver, t)
        // - nếu new => push vào self.new_time_points
    }

    pub fn apply_model(&mut self, model: &impl satcoder::prelude::SymbolicModel<L>) {
        // === COPY đoạn "update times" từ nhánh SAT(model) vào đây ===
        // - update incumbent_idx
        // - update touched_intervals (thêm prev_visit nếu cần)
    }

    pub fn refine_after_sat(
        &mut self,
        solver: &mut impl SatInstance<L>,
        problem: &Problem,
        stats: &mut SolveStats,
        mut on_action: impl FnMut(SolverAction),
    ) -> IterationType {
        // === COPY nguyên block conflict detection:
        // - travel conflict loop
        // - resource conflict loop + add clauses + add new timepoints
        // - return IterationType (Solution/Conflict...)
        unimplemented!()
    }

    pub fn drain_new_timepoints_and_encode_cost(
        &mut self,
        solver: &mut impl SatInstance<L>,
        problem: &Problem,
        delay_cost_type: DelayCostType,
        mode: CostMode,
        mut on_cost: impl FnMut(CostTerm<L>),
    ) {
        // === COPY nguyên vòng for new_time_points.drain(..)
        // Điểm khác:
        // - Ladder: mỗi khi tạo next_cost_var => on_cost(CostTerm{var: next_cost_var, weight:1})
        // - CostTree: trong notify_vars(weight, cost_var) => on_cost(CostTerm{var: cost_var, weight})
        unimplemented!()
    }
}

pub fn do_output_stats<L: satcoder::Lit>(
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
    output_stats("num_conflicts".to_string(), stats.n_conflict.into());
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
            .unwrap_or(0)
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

pub fn extract_solution<L: satcoder::Lit>(
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
