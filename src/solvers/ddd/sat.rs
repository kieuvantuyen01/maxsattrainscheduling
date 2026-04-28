use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    sync::mpsc,
    time::{Duration, Instant},
};

use crate::{
    debug::{DebugInfo, ResourceInterval, SolverAction},
    problem::{DelayCostType, Problem},
    solvers::{heuristic, value_trace::ValueTrace},
};
use rustsat::{
    encodings::{
        pb::{BoundUpper, BoundUpperIncremental, Encode as PbEncode, GeneralizedTotalizer},
        CollectClauses,
    },
    instances::ManageVars,
    solvers::{
        Interrupt as RsInterrupt, InterruptSolver as RsInterruptSolver, Solve as RsSolve,
        SolveIncremental as RsSolveIncremental, SolveStats as RsSolveStats,
        SolverResult as RsSolverResult,
    },
    types::{
        Assignment as RsAssignment, Clause as RsClause, Lit as RsLit, TernaryVal as RsTernaryVal,
        Var as RsVar,
    },
    OutOfMemory as RsOutOfMemory,
};
use rustsat_glucose::core::Glucose as RsGlucose;
use satcoder::{
    prelude::SymbolicModel, Bool, SatInstance, SatModel, SatResult, SatResultWithCore, SatSolver,
    SatSolverWithCore,
};
use typed_index_collections::TiVec;

use super::{
    common::{do_output_stats, extract_solution, IterationType, Occ, SolveStats, VisitId},
    costtree::CostTree,
};
use crate::solvers::SolverError;

type NativeLit = RsLit;

#[derive(Clone, Copy, Debug)]
pub enum SatBoundMode {
    AddClauses,
    Assumptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatPrecEncoding {
    Plain,
    Scl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatSearchMode {
    UbSearch,
    Invalid,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ResourceCliqueRowKey {
    // (visit_id, start, next_incumbent_time). Includes next_incumbent so that
    // when the next-visit's incumbent shifts (changing the clique's min-end
    // and thus the AMO's tau+1), the key changes and a fresh tight constraint
    // is added. If state is unchanged, the existing AMO is still valid → skip.
    members: Vec<(VisitId, i32, i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatObjectiveEncoding {
    Scpb,
    IncrementalTotalizer,
    BitTotalizer,
}

#[derive(Clone, Copy, Debug)]
pub struct SatDddSettings {
    pub use_precedence_graph: bool,
}

impl Default for SatDddSettings {
    fn default() -> Self {
        Self {
            use_precedence_graph: true,
        }
    }
}

#[derive(Default)]
struct NativeSolver {
    inner: RsGlucose,
    next_var: u32,
    solve_timeout: Option<Duration>,
    was_interrupted: bool,
}

impl NativeSolver {
    fn new() -> Self {
        Self::default()
    }

    fn reserve_var(&mut self, var: RsVar) {
        self.inner.reserve(var).expect("glucose reserve failed");
        let next_free = var.idx32() + 1;
        if next_free > self.next_var {
            self.next_var = next_free;
        }
    }

    fn reserve_clause(&mut self, clause: &RsClause) {
        if let Some(max_var) = AsRef::<[RsLit]>::as_ref(clause)
            .iter()
            .map(|lit| lit.var())
            .max()
        {
            self.reserve_var(max_var);
        }
    }

    fn set_solve_timeout(&mut self, timeout: Option<Duration>) {
        self.solve_timeout = timeout;
    }

    fn take_interrupted(&mut self) -> bool {
        std::mem::take(&mut self.was_interrupted)
    }
}

impl std::fmt::Debug for NativeSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RustSATGlucoseNative")
    }
}

#[derive(Debug)]
struct NativeModel(RsAssignment);

impl SatModel for NativeModel {
    type Lit = NativeLit;

    fn lit_value(&self, l: &Self::Lit) -> bool {
        matches!(self.0.lit_value(*l), RsTernaryVal::True)
    }
}

enum NativeSolveResult {
    Sat(Box<dyn SatModel<Lit = NativeLit>>),
    Unsat(Box<[NativeLit]>),
    Interrupted,
}

impl SatInstance<NativeLit> for NativeSolver {
    fn new_var(&mut self) -> Bool<NativeLit> {
        let v = RsVar::new(self.next_var);
        self.next_var += 1;
        self.inner.reserve(v).expect("glucose reserve failed");
        Bool::Lit(v.pos_lit())
    }

    fn add_clause<IL: Into<Bool<NativeLit>>, I: IntoIterator<Item = IL>>(&mut self, clause: I) {
        let mut lits: Vec<NativeLit> = Vec::new();
        for b in clause {
            match b.into() {
                Bool::Const(true) => return,
                Bool::Const(false) => {}
                Bool::Lit(l) => lits.push(l),
            }
        }
        let cl: RsClause = lits.into_iter().collect();
        self.reserve_clause(&cl);
        self.inner
            .add_clause_ref(&cl)
            .expect("glucose add_clause_ref failed");
    }
}

impl NativeSolver {
    fn solve_with_assumptions_owned(
        &mut self,
        assumptions: impl IntoIterator<Item = Bool<NativeLit>>,
    ) -> NativeSolveResult {
        self.was_interrupted = false;
        let mut assumps: Vec<NativeLit> = Vec::new();
        for a in assumptions {
            match a {
                Bool::Const(true) => {}
                Bool::Const(false) => panic!("unsat assumption"),
                Bool::Lit(l) => assumps.push(l),
            }
        }

        let timeout_guard = self.solve_timeout.map(|limit| {
            let interrupter = self.inner.interrupter();
            let (done_tx, done_rx) = mpsc::channel();
            let join_handle = std::thread::spawn(move || {
                if done_rx.recv_timeout(limit).is_err() {
                    interrupter.interrupt();
                }
            });
            (done_tx, join_handle)
        });

        let result = self.inner.solve_assumps(&assumps);

        if let Some((done_tx, join_handle)) = timeout_guard {
            let _ = done_tx.send(());
            let _ = join_handle.join();
        }

        match result {
            Ok(RsSolverResult::Sat) => {
                let model = self
                    .inner
                    .full_solution()
                    .expect("glucose: failed to get full solution");
                NativeSolveResult::Sat(Box::new(NativeModel(model)))
            }
            Ok(RsSolverResult::Unsat) => {
                let core = self.inner.core().unwrap_or_default();
                NativeSolveResult::Unsat(core.into_boxed_slice())
            }
            Ok(RsSolverResult::Interrupted) => {
                self.was_interrupted = true;
                NativeSolveResult::Interrupted
            }
            Err(e) => panic!("glucose solve error: {}", e),
        }
    }
}

impl SatSolverWithCore for NativeSolver {
    type Lit = NativeLit;

    fn solve_with_assumptions<'a>(
        &'a mut self,
        assumptions: impl IntoIterator<Item = Bool<Self::Lit>>,
    ) -> SatResultWithCore<'a, Self::Lit> {
        match self.solve_with_assumptions_owned(assumptions) {
            NativeSolveResult::Sat(model) => SatResultWithCore::Sat(model),
            NativeSolveResult::Unsat(core) => SatResultWithCore::Unsat(core),
            NativeSolveResult::Interrupted => {
                self.was_interrupted = true;
                SatResultWithCore::Unsat(Box::new([]))
            }
        }
    }
}

impl SatSolver for NativeSolver {
    type Lit = NativeLit;

    fn solve<'a>(&'a mut self) -> SatResult<'a, Self::Lit> {
        match self.solve_with_assumptions(std::iter::empty()) {
            SatResultWithCore::Sat(m) => SatResult::Sat(m),
            SatResultWithCore::Unsat(_) => SatResult::Unsat,
        }
    }
}

struct NativeClauseCollector<'a> {
    inner: &'a mut RsGlucose,
}

impl CollectClauses for NativeClauseCollector<'_> {
    fn n_clauses(&self) -> usize {
        RsSolveStats::n_clauses(&*self.inner)
    }

    fn extend_clauses<T>(&mut self, cl_iter: T) -> Result<(), RsOutOfMemory>
    where
        T: IntoIterator<Item = RsClause>,
    {
        for cl in cl_iter {
            if let Some(max_var) = AsRef::<[RsLit]>::as_ref(&cl)
                .iter()
                .map(|lit| lit.var())
                .max()
            {
                self.inner
                    .reserve(max_var)
                    .map_err(|_| RsOutOfMemory::ExternalApi)?;
            }
            self.inner
                .add_clause_ref(&cl)
                .map_err(|_| RsOutOfMemory::ExternalApi)?;
        }
        Ok(())
    }
}

struct NativeVarManager<'a> {
    next_var: &'a mut u32,
}

impl ManageVars for NativeVarManager<'_> {
    fn new_var(&mut self) -> RsVar {
        let v = RsVar::new(*self.next_var);
        *self.next_var += 1;
        v
    }

    fn max_var(&self) -> Option<RsVar> {
        if *self.next_var == 0 {
            None
        } else {
            Some(RsVar::new(*self.next_var - 1))
        }
    }

    fn increase_next_free(&mut self, v: RsVar) -> bool {
        if v.idx32() > *self.next_var {
            *self.next_var = v.idx32();
            return true;
        }
        false
    }

    fn combine(&mut self, other: Self) {
        let other_next = *other.next_var;
        if other_next > *self.next_var {
            *self.next_var = other_next;
        }
    }

    fn n_used(&self) -> u32 {
        *self.next_var
    }

    fn forget_from(&mut self, min_var: RsVar) {
        *self.next_var = std::cmp::min(*self.next_var, min_var.idx32());
    }
}

pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::AddClauses,
        output_stats,
    )
}

pub fn solve_with_encoding<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_encoding_and_settings(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        SatDddSettings::default(),
        output_stats,
    )
}

pub fn solve_with_encoding_and_settings<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    settings: SatDddSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let mode = SatBoundMode::Assumptions;
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        settings,
        mode,
        SatPrecEncoding::Plain,
        SatSearchMode::UbSearch,
        |_| {},
        output_stats,
    )
}

pub fn solve_incremental<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_mode(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::Assumptions,
        SatPrecEncoding::Plain,
        SatSearchMode::Invalid,
        |_| {},
        output_stats,
    )
}

pub fn solve_incremental_with_encoding<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_incremental_with_encoding_and_settings(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        SatDddSettings::default(),
        output_stats,
    )
}

pub fn solve_incremental_with_encoding_and_settings<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    settings: SatDddSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        settings,
        SatBoundMode::Assumptions,
        SatPrecEncoding::Plain,
        SatSearchMode::Invalid,
        |_| {},
        output_stats,
    )
}

pub fn solve_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode_scl(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::AddClauses,
        SatSearchMode::UbSearch,
        output_stats,
    )
}

pub fn solve_scl_with_encoding<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_scl_with_encoding_and_settings(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        SatDddSettings::default(),
        output_stats,
    )
}

pub fn solve_scl_with_encoding_and_settings<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    settings: SatDddSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let mode = SatBoundMode::Assumptions;
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        settings,
        mode,
        SatPrecEncoding::Scl,
        SatSearchMode::UbSearch,
        |_| {},
        output_stats,
    )
}

pub fn solve_incremental_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode_scl(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::Assumptions,
        SatSearchMode::UbSearch,
        output_stats,
    )
}

pub fn solve_incremental_scl_with_encoding<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_incremental_scl_with_encoding_and_settings(
        mk_env,
        _solver,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        SatDddSettings::default(),
        output_stats,
    )
}

pub fn solve_incremental_scl_with_encoding_and_settings<
    L: satcoder::Lit + Copy + std::fmt::Debug,
>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    settings: SatDddSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        encoding,
        settings,
        SatBoundMode::Assumptions,
        SatPrecEncoding::Scl,
        SatSearchMode::UbSearch,
        |_| {},
        output_stats,
    )
}

pub fn solve_with_mode<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        SatObjectiveEncoding::Scpb,
        SatDddSettings::default(),
        mode,
        SatPrecEncoding::Plain,
        SatSearchMode::UbSearch,
        |_| {},
        output_stats,
    )
}

pub fn solve_with_mode_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    search: SatSearchMode,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        SatObjectiveEncoding::Scpb,
        SatDddSettings::default(),
        mode,
        SatPrecEncoding::Scl,
        search,
        |_| {},
        output_stats,
    )
}

thread_local! { pub static WATCH : RefCell<Option<(usize,usize)>> = RefCell::new(None); }

fn add_guarded_clause(
    solver: &mut NativeSolver,
    gate: Option<Bool<NativeLit>>,
    clause: impl IntoIterator<Item = Bool<NativeLit>>,
) {
    let mut lits: Vec<Bool<NativeLit>> = clause.into_iter().collect();
    if let Some(sel) = gate {
        lits.insert(0, !sel);
    }
    SatInstance::add_clause(solver, lits);
}

fn encode_scpb_leq(
    solver: &mut NativeSolver,
    terms: &[(NativeLit, usize)],
    bound: usize,
    gate: Option<Bool<NativeLit>>,
) {
    if terms.is_empty() {
        return;
    }

    let mut bounded_terms: Vec<(Bool<NativeLit>, usize)> = Vec::with_capacity(terms.len());
    for &(lit, weight) in terms {
        if weight == 0 {
            continue;
        }

        let term = Bool::from_lit(lit);
        if weight > bound {
            add_guarded_clause(solver, gate, [!term]);
        } else {
            bounded_terms.push((term, weight));
        }
    }

    if bounded_terms.is_empty() {
        return;
    }

    if bound == 0 {
        for (term, _) in bounded_terms {
            add_guarded_clause(solver, gate, [!term]);
        }
        return;
    }

    if bounded_terms.len() == 1 {
        return;
    }

    let n_terms = bounded_terms.len();
    let mut counters: Vec<Vec<Bool<NativeLit>>> = Vec::with_capacity(n_terms - 1);
    let mut prefix_weight = 0usize;

    for &(_, weight) in bounded_terms.iter().take(n_terms - 1) {
        prefix_weight = (prefix_weight + weight).min(bound);
        let mut row = Vec::with_capacity(prefix_weight);
        for _ in 0..prefix_weight {
            row.push(SatInstance::new_var(solver));
        }
        counters.push(row);
    }

    // SCPB_<=k clauses (1), (2), and (3) on the first n-1 weighted terms.
    for term_idx in 0..(n_terms - 1) {
        let (term, weight) = bounded_terms[term_idx];
        let row = &counters[term_idx];

        for bit_idx in 0..weight.min(row.len()) {
            add_guarded_clause(solver, gate, [!term, row[bit_idx]]);
        }

        if term_idx == 0 {
            continue;
        }

        let prev_row = &counters[term_idx - 1];
        for bit_idx in 0..prev_row.len() {
            add_guarded_clause(solver, gate, [!prev_row[bit_idx], row[bit_idx]]);

            let target_idx = bit_idx + weight;
            if target_idx < row.len() {
                add_guarded_clause(solver, gate, [!term, !prev_row[bit_idx], row[target_idx]]);
            }
        }
    }

    // SCPB_<=k clause (8) on the remaining term against the previous counter row.
    for term_idx in 1..n_terms {
        let (term, weight) = bounded_terms[term_idx];
        let prev_row = &counters[term_idx - 1];
        let threshold_bit = bound + 1 - weight;
        if threshold_bit <= prev_row.len() {
            add_guarded_clause(solver, gate, [!term, !prev_row[threshold_bit - 1]]);
        }
    }
}

fn encode_exact_unary_counter_limited(
    solver: &mut NativeSolver,
    inputs: &[Bool<NativeLit>],
    max_threshold: usize,
) -> Vec<Bool<NativeLit>> {
    if max_threshold == 0 {
        return Vec::new();
    }

    let mut fixed_true = 0usize;
    let mut variable_inputs = Vec::new();

    for input in inputs.iter().copied() {
        match input {
            Bool::Const(true) => fixed_true += 1,
            Bool::Const(false) => {}
            Bool::Lit(_) => variable_inputs.push(input),
        }
    }

    let max_variable_threshold = max_threshold.saturating_sub(fixed_true);
    let variable_count =
        encode_balanced_totalizer_limited(solver, &variable_inputs, max_variable_threshold);

    let mut outputs = Vec::with_capacity(max_threshold);
    for threshold in 1..=max_threshold {
        if threshold <= fixed_true {
            outputs.push(true.into());
        } else {
            let variable_threshold = threshold - fixed_true;
            if variable_threshold <= variable_count.len() {
                outputs.push(variable_count[variable_threshold - 1]);
            } else {
                outputs.push(false.into());
            }
        }
    }
    outputs
}

fn unary_counter_at(counter: &[Bool<NativeLit>], threshold: usize) -> Bool<NativeLit> {
    if threshold == 0 {
        true.into()
    } else {
        counter.get(threshold - 1).copied().unwrap_or(false.into())
    }
}

fn encode_totalizer_merge_limited(
    solver: &mut NativeSolver,
    left: &[Bool<NativeLit>],
    right: &[Bool<NativeLit>],
    limit: usize,
) -> Vec<Bool<NativeLit>> {
    let out_len = limit.min(left.len() + right.len());
    if out_len == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(out_len);
    for _ in 0..out_len {
        out.push(SatInstance::new_var(solver));
    }

    for i in 1..=left.len().min(out_len) {
        SatInstance::add_clause(solver, vec![!left[i - 1], out[i - 1]]);
    }
    for j in 1..=right.len().min(out_len) {
        SatInstance::add_clause(solver, vec![!right[j - 1], out[j - 1]]);
    }
    for i in 1..=left.len() {
        for j in 1..=right.len() {
            let threshold = i + j;
            if threshold <= out_len {
                SatInstance::add_clause(
                    solver,
                    vec![!left[i - 1], !right[j - 1], out[threshold - 1]],
                );
            }
        }
    }

    for threshold in 1..=out_len {
        for left_max in 0..threshold {
            let right_max = threshold - 1 - left_max;
            SatInstance::add_clause(
                solver,
                vec![
                    !out[threshold - 1],
                    unary_counter_at(left, left_max + 1),
                    unary_counter_at(right, right_max + 1),
                ],
            );
        }
    }

    for threshold in 2..=out_len {
        SatInstance::add_clause(solver, vec![!out[threshold - 1], out[threshold - 2]]);
    }

    out
}

fn encode_balanced_totalizer_limited(
    solver: &mut NativeSolver,
    inputs: &[Bool<NativeLit>],
    limit: usize,
) -> Vec<Bool<NativeLit>> {
    let limit = limit.min(inputs.len());
    if limit == 0 || inputs.is_empty() {
        return Vec::new();
    }
    if inputs.len() == 1 {
        return vec![inputs[0]];
    }

    let mid = inputs.len() / 2;
    let left = encode_balanced_totalizer_limited(solver, &inputs[..mid], limit);
    let right = encode_balanced_totalizer_limited(solver, &inputs[mid..], limit);
    encode_totalizer_merge_limited(solver, &left, &right, limit)
}

fn encode_parity_from_unary_counter(
    solver: &mut NativeSolver,
    ge: &[Bool<NativeLit>],
) -> Bool<NativeLit> {
    if ge.is_empty() {
        return false.into();
    }
    if ge.len() == 1 {
        return ge[0];
    }

    let parity = SatInstance::new_var(solver);
    for count in 0..=ge.len() {
        let expected_parity = if count % 2 == 1 { parity } else { !parity };
        let clause = if count == 0 {
            vec![ge[0], expected_parity]
        } else if count == ge.len() {
            vec![!ge[count - 1], expected_parity]
        } else {
            vec![!ge[count - 1], ge[count], expected_parity]
        };
        SatInstance::add_clause(solver, clause);
    }

    parity
}

fn shifted_bound(bound: usize, bit_idx: usize) -> usize {
    if bit_idx >= usize::BITS as usize {
        0
    } else {
        bound >> bit_idx
    }
}

fn encode_bit_totalizer_sum_bits_with_overflow(
    solver: &mut NativeSolver,
    terms: &[(NativeLit, usize)],
    bound: Option<usize>,
) -> (Vec<Bool<NativeLit>>, Vec<Bool<NativeLit>>) {
    let mut buckets: Vec<Vec<Bool<NativeLit>>> = Vec::new();
    for &(lit, weight) in terms {
        if weight == 0 {
            continue;
        }

        let term = Bool::from_lit(lit);
        let mut remaining = weight;
        let mut bit_idx = 0usize;
        while remaining > 0 {
            if remaining & 1 == 1 {
                if buckets.len() <= bit_idx {
                    buckets.resize_with(bit_idx + 1, Vec::new);
                }
                buckets[bit_idx].push(term);
            }
            remaining >>= 1;
            bit_idx += 1;
        }
    }

    let mut sum_bits = Vec::new();
    let mut overflow_lits = Vec::new();
    let mut carry: Vec<Bool<NativeLit>> = Vec::new();
    let mut bit_idx = 0usize;

    while bit_idx < buckets.len() || !carry.is_empty() {
        let mut units = Vec::new();
        if bit_idx < buckets.len() {
            units.extend(buckets[bit_idx].iter().copied());
        }
        units.extend(std::mem::take(&mut carry));

        if units.is_empty() {
            sum_bits.push(false.into());
        } else {
            let cap_count = bound.map(|b| shifted_bound(b, bit_idx));
            let max_threshold = cap_count
                .and_then(|cap| cap.checked_add(1))
                .unwrap_or(units.len())
                .min(units.len());
            let ge = encode_exact_unary_counter_limited(solver, &units, max_threshold);
            if let Some(cap) = cap_count {
                if let Some(overflow_threshold) = cap.checked_add(1) {
                    if overflow_threshold <= ge.len() {
                        overflow_lits.push(ge[overflow_threshold - 1]);
                    }
                }
            }
            sum_bits.push(encode_parity_from_unary_counter(solver, &ge));
            let max_carry = cap_count
                .map(|cap| (cap / 2).min(ge.len() / 2))
                .unwrap_or(ge.len() / 2);
            carry = (1..=max_carry).map(|idx| ge[(2 * idx) - 1]).collect();
        }

        bit_idx += 1;
    }

    (sum_bits, overflow_lits)
}

fn usize_bit(value: usize, bit_idx: usize) -> bool {
    bit_idx < usize::BITS as usize && ((value >> bit_idx) & 1) == 1
}

fn encode_binary_leq_constant(
    solver: &mut NativeSolver,
    bits: &[Bool<NativeLit>],
    bound: usize,
    gate: Option<Bool<NativeLit>>,
) {
    for bit_idx in 0..bits.len() {
        if usize_bit(bound, bit_idx) {
            continue;
        }

        let mut clause = Vec::with_capacity(bits.len() - bit_idx);
        clause.push(!bits[bit_idx]);
        for higher_idx in (bit_idx + 1)..bits.len() {
            if usize_bit(bound, higher_idx) {
                clause.push(!bits[higher_idx]);
            } else {
                clause.push(bits[higher_idx]);
            }
        }
        add_guarded_clause(solver, gate, clause);
    }
}

#[derive(Default)]
struct BitTotalizerBoundNetwork {
    encoded_terms: usize,
    sum_bits: Vec<Bool<NativeLit>>,
    overflow_lits: Vec<Bool<NativeLit>>,
}

#[derive(Default)]
struct BitTotalizerObjective {
    terms: Vec<(NativeLit, usize)>,
    total_weight: usize,
    bound_networks: HashMap<usize, BitTotalizerBoundNetwork>,
    addclauses_bounds: HashMap<usize, usize>,
    assumption_bounds: HashMap<usize, (usize, Bool<NativeLit>)>,
}

impl BitTotalizerObjective {
    fn add_term(&mut self, lit: NativeLit, weight: usize) {
        if weight == 0 {
            return;
        }
        self.terms.push((lit, weight));
        self.total_weight = self.total_weight.saturating_add(weight);
    }

    fn term_count(&self) -> usize {
        self.terms.len()
    }

    fn ensure_bound_network(&mut self, solver: &mut NativeSolver, bound: usize) {
        if self
            .bound_networks
            .get(&bound)
            .map(|network| network.encoded_terms == self.terms.len())
            .unwrap_or(false)
        {
            return;
        }

        let (sum_bits, overflow_lits) =
            encode_bit_totalizer_sum_bits_with_overflow(solver, &self.terms, Some(bound));
        self.bound_networks.insert(
            bound,
            BitTotalizerBoundNetwork {
                encoded_terms: self.terms.len(),
                sum_bits,
                overflow_lits,
            },
        );
    }

    fn encode_leq(
        &mut self,
        solver: &mut NativeSolver,
        bound: usize,
        gate: Option<Bool<NativeLit>>,
    ) {
        self.ensure_bound_network(solver, bound);
        let network = self
            .bound_networks
            .get(&bound)
            .expect("bit totalizer bound network was just encoded");
        for &overflow in &network.overflow_lits {
            add_guarded_clause(solver, gate, [!overflow]);
        }
        encode_binary_leq_constant(solver, &network.sum_bits, bound, gate);
    }
}

fn inject_solution_timepoints_sat<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    train_visit_ids: &[Vec<VisitId>],
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    sol: &[Vec<i32>],
) {
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            let t = sol[train_idx][visit_idx];
            let vid = train_visit_ids[train_idx][visit_idx];
            let (v, is_new) = occupations[vid].time_point(solver, t);
            if is_new {
                new_time_points.push((vid, v, t));
            }
        }
    }
}

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
    prec: SatPrecEncoding,
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
            prec,
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
    prec: SatPrecEncoding,
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
        prec,
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
            prec,
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
            prec,
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
}

pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    debug_out: impl Fn(DebugInfo),
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        SatObjectiveEncoding::Scpb,
        SatDddSettings::default(),
        SatBoundMode::AddClauses,
        SatPrecEncoding::Plain,
        SatSearchMode::UbSearch,
        debug_out,
        output_stats,
    )
}

pub fn solve_debug_with_mode<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    _solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    prec: SatPrecEncoding,
    search: SatSearchMode,
    debug_out: impl Fn(DebugInfo),
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_native_debug_with_mode(
        mk_env,
        problem,
        timeout,
        delay_cost_type,
        SatObjectiveEncoding::Scpb,
        SatDddSettings::default(),
        mode,
        prec,
        search,
        debug_out,
        output_stats,
    )
}

fn solve_native_debug_with_mode(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    encoding: SatObjectiveEncoding,
    settings: SatDddSettings,
    mode: SatBoundMode,
    prec: SatPrecEncoding,
    search: SatSearchMode,
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let _p = hprof::enter("sat_solver");
    let search_label = match search {
        SatSearchMode::UbSearch => "ub_search",
        SatSearchMode::Invalid => "invalid",
    };
    println!("SAT search mode: {}", search_label);

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();
    let mut solver = NativeSolver::new();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut train_visit_ids: Vec<Vec<VisitId>> = vec![Vec::new(); problem.trains.len()];
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<NativeLit>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points: Vec<(VisitId, Bool<NativeLit>, i32)> = Vec::new();
    let effective_earliest = settings
        .use_precedence_graph
        .then(|| compute_effective_earliest(problem));

    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0usize;
    let mut n_conflict_constraints = 0usize;

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));
            train_visit_ids[train_idx].push(visit_id);
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

            const SEED_COST_THRESHOLDS: bool = true;
            if SEED_COST_THRESHOLDS {
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
    }

    let mut best_sol: Option<(i32, Vec<Vec<i32>>)> = None;
    let mut lower_bound: i32 = 0;
    let mut upper_bound: Option<i32> = None;
    let use_cont_fixed_query = matches!(delay_cost_type, DelayCostType::Continuous);
    let mut cont_active_query_bound: Option<i32> = None;
    let mut value_trace = ValueTrace::default();
    let mut logged_incumbent: Option<i32> = None;
    let mut logged_lower_bound: Option<i32> = None;
    let trace_bound_queries = problem.name == "instances_original/InstanceA1.txt";

    let mut scpb_terms: Vec<(NativeLit, usize)> = Vec::new();
    let mut scpb_total_weight = 0usize;
    let mut scpb_addclauses_bounds: HashMap<usize, usize> = HashMap::new();
    let mut scpb_assumption_bounds: HashMap<usize, (usize, Bool<NativeLit>)> = HashMap::new();
    let mut budget_gte = GeneralizedTotalizer::default();
    let mut last_added_bound: Option<usize> = None;
    let mut bit_totalizer = BitTotalizerObjective::default();

    let mut added_resource_clique_rows: HashSet<ResourceCliqueRowKey> = HashSet::new();
    let mut fixed_prec_rows: HashSet<(VisitId, i32)> = HashSet::new();

    const SEED_PRECEDENCE_FROM_EARLIEST: bool = true;
    if SEED_PRECEDENCE_FROM_EARLIEST {
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
                    prec,
                );
            } else if prec == SatPrecEncoding::Scl {
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
                    prec,
                );
            }
        }
    }

    // Seed pairwise AMO constraints for visit pairs whose earliest occupation
    // intervals overlap. These conflicts are guaranteed at earliest — DDD would
    // rediscover them in the first few iterations anyway. Seeding upfront lets
    // SAT solver reason about them from iteration 1, often saving many DDD
    // iterations on large instances where the LB-UB gap is tight.
    //
    // Per pair: build two `active_i(tau_plus_1)` aux vars (3 Tseitin clauses each)
    // and add one AMO clause `!active_v1 OR !active_v2`. Total ≈ 7 clauses + 2
    // aux vars per seeded pair. For typical instances this is ~5-20k extra
    // clauses; SAT solver handles this easily in exchange for faster convergence.
    const SEED_RESOURCE_CONFLICTS: bool = true;
    // Only seed overlaps ≥ 180s — aligns with InfiniteSteps180 step boundaries.
    // For finer objectives (cont, infsteps60), smaller conflicts may also matter;
    // tune lower if benchmarks show more seeding helps.
    const MIN_OVERLAP_FOR_SEED: i32 = 180;
    if SEED_RESOURCE_CONFLICTS {
        let mut seed_active_cache: HashMap<(VisitId, i32), Bool<NativeLit>> = HashMap::new();
        let mut n_seeded = 0usize;
        let mut seen_resource_pairs: HashSet<(usize, usize)> = HashSet::new();

        // Helper: get earliest start + earliest end of a visit's occupation.
        let earliest_occupation = |visit_id: VisitId,
                                   occupations: &TiVec<VisitId, Occ<NativeLit>>|
         -> (i32, i32) {
            let (train_idx, visit_idx) = visits[visit_id];
            let e_start = occupations[visit_id].delays[0].1;
            let e_end = if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                let next_id: VisitId = (usize::from(visit_id) + 1).into();
                occupations[next_id].delays[0].1
            } else {
                e_start + problem.trains[train_idx].visits[visit_idx].travel_time
            };
            (e_start, e_end)
        };

        let mut try_seed_pair =
            |v1: VisitId,
             v2: VisitId,
             solver: &mut NativeSolver,
             occupations: &mut TiVec<VisitId, Occ<NativeLit>>,
             new_time_points: &mut Vec<(VisitId, Bool<NativeLit>, i32)>,
             fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
             seed_active_cache: &mut HashMap<(VisitId, i32), Bool<NativeLit>>,
             n_seeded: &mut usize| {
                let (t1_idx, _) = visits[v1];
                let (t2_idx, _) = visits[v2];
                if t1_idx == t2_idx {
                    return;
                }

                let (e1_start, e1_end) = earliest_occupation(v1, occupations);
                let (e2_start, e2_end) = earliest_occupation(v2, occupations);

                if e1_end <= e1_start || e2_end <= e2_start {
                    return;
                }

                // Overlap at earliest: intervals [e1_start, e1_end) ∩ [e2_start, e2_end)
                let overlap_start = e1_start.max(e2_start);
                let overlap_end = e1_end.min(e2_end);
                if overlap_start + MIN_OVERLAP_FOR_SEED > overlap_end {
                    return;
                }

                // tau_plus_1 = min end — same as runtime AMO semantics.
                let tau_plus_1 = overlap_end;

                let active_v1 = build_active_lit(
                    solver,
                    problem,
                    &visits,
                    occupations,
                    new_time_points,
                    fixed_prec_rows,
                    seed_active_cache,
                    prec,
                    v1,
                    tau_plus_1,
                );
                let active_v2 = build_active_lit(
                    solver,
                    problem,
                    &visits,
                    occupations,
                    new_time_points,
                    fixed_prec_rows,
                    seed_active_cache,
                    prec,
                    v2,
                    tau_plus_1,
                );

                solver.add_clause(vec![!active_v1, !active_v2]);
                *n_seeded += 1;
            };

        // Iterate over unique conflicting resource pairs.
        let conflicts_snapshot: Vec<(usize, Vec<usize>)> = conflicts
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        for (res_a, others) in &conflicts_snapshot {
            for &res_b in others {
                if *res_a > res_b {
                    continue;
                }
                if !seen_resource_pairs.insert((*res_a, res_b)) {
                    continue;
                }
                if *res_a >= resource_visits.len() || res_b >= resource_visits.len() {
                    continue;
                }

                if *res_a == res_b {
                    let visits_list = resource_visits[*res_a].clone();
                    for i in 0..visits_list.len() {
                        for j in (i + 1)..visits_list.len() {
                            try_seed_pair(
                                visits_list[i],
                                visits_list[j],
                                &mut solver,
                                &mut occupations,
                                &mut new_time_points,
                                &mut fixed_prec_rows,
                                &mut seed_active_cache,
                                &mut n_seeded,
                            );
                        }
                    }
                } else {
                    let visits_a = resource_visits[*res_a].clone();
                    let visits_b = resource_visits[res_b].clone();
                    for &v1 in &visits_a {
                        for &v2 in &visits_b {
                            try_seed_pair(
                                v1,
                                v2,
                                &mut solver,
                                &mut occupations,
                                &mut new_time_points,
                                &mut fixed_prec_rows,
                                &mut seed_active_cache,
                                &mut n_seeded,
                            );
                        }
                    }
                }
            }
        }

        println!(
            "SAT resource-conflict seeding: {} pairwise AMOs added",
            n_seeded
        );
    }

    const USE_INITIAL_HEURISTIC_UB_ONLY: bool = true;
    if USE_INITIAL_HEURISTIC_UB_ONLY {
        if let Some((ub_cost, ub_sol)) =
            compute_initial_heuristic_upper_bound(&mk_env, problem, delay_cost_type, &occupations)?
        {
            println!("SAT initial heuristic UB={}", ub_cost);
            if trace_bound_queries {
                eprintln!(
                    "[SAT-BOUND-TRACE] instance={} event=initial_heuristic_ub lb={} ub={} cost={} search={:?} mode={:?} encoding={:?}",
                    problem.name,
                    lower_bound,
                    ub_cost,
                    ub_cost,
                    search,
                    mode,
                    encoding
                );
            }
            best_sol = Some((ub_cost, ub_sol));
            value_trace.initial_incumbent(start_time, ub_cost, lower_bound, None);
            logged_incumbent = Some(ub_cost);
            if search == SatSearchMode::UbSearch {
                upper_bound = Some(ub_cost - 1);
            }
        }
    }

    let mut iteration: usize = 1;
    let mut is_sat: bool = true;
    let mut invalid_clause: Vec<Bool<NativeLit>> = Vec::new();

    loop {
        let mut bound_assumptions: Vec<Bool<NativeLit>> = Vec::new();
        let mut bound_used: Option<i32> = None;
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_sol.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX);
            let lb = lower_bound;
            println!("TIMEOUT LB={} UB={}", lb, ub);

            value_trace.timeout(start_time, lb, best_sol.as_ref().map(|(c, _)| *c), Some(iteration as i32));
            value_trace.emit(&mut output_stats, best_sol.as_ref().map(|(c, _)| *c));
            do_output_stats(
                &mut output_stats,
                iteration,
                &iteration_types,
                &stats,
                &occupations,
                start_time,
                solver_time,
                lb,
                ub,
            );
            return Err(SolverError::Timeout);
        }

        if is_sat {
            let mut found_travel_time_conflict = false;
            let mut found_resource_conflict = false;

            for visit_id in touched_intervals.iter().copied() {
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

                        let mut debug_actions = Vec::new();
                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));
                        debug_out(DebugInfo {
                            iteration,
                            actions: debug_actions,
                            solution: extract_solution(problem, &occupations),
                        });

                        let in_var = v1.delays[v1.incumbent_idx].0;
                        let in_t = v1.incumbent_time();
                        let added_prec = add_fixed_precedence_row(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            visit_id,
                            in_var,
                            in_t,
                            prec,
                        );
                        if trace_bound_queries {
                            if let Some((next_visit, _, req_t)) = added_prec {
                                eprintln!(
                                    "[SAT-BOUND-TRACE] instance={} event=precedence_row_add iter={} source=travel_conflict visit_id={} train_idx={} visit_idx={} in_t={} next_visit_id={} req_t={} fixed_prec_rows={} pending_new_time_points={}",
                                    problem.name,
                                    iteration,
                                    visit_id.0,
                                    train_idx,
                                    visit_idx,
                                    in_t,
                                    next_visit.0,
                                    req_t,
                                    fixed_prec_rows.len(),
                                    new_time_points.len()
                                );
                            }
                        }
                        stats.n_travel += 1;
                    }
                }
            }

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

            // BTreeSet for deterministic iteration order across runs (reproducibility).
            let mut relevant_resource_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
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
            let mut active_lit_cache: HashMap<(VisitId, i32), Bool<NativeLit>> = HashMap::new();

            // Collect all clique candidates across resource pairs first, then
            // process them in severity order with a per-iteration budget. This
            // ensures the most critical conflicts are resolved first even when
            // the budget is hit.
            let mut clique_candidates: Vec<(i64, Vec<ActiveInterval>, i32, usize, usize)> =
                Vec::new();

            for (resource_a, resource_b) in relevant_resource_pairs.into_iter() {
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
                        let next_visit: Option<VisitId> =
                            if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                Some((usize::from(visit_id) + 1).into())
                            } else {
                                None
                            };
                        let end = next_visit
                            .map(|nx| occupations[nx].incumbent_time())
                            .unwrap_or(
                                start + problem.trains[train_idx].visits[visit_idx].travel_time,
                            );
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

            // Sort by severity descending — the most "dangerous" cliques (largest
            // with longest overlaps) are processed first each iteration. This
            // pushes the LB up faster and closes the LB-UB gap more aggressively.
            clique_candidates.sort_by(|a, b| b.0.cmp(&a.0));

            // Per-iteration budget: cap the number of cliques processed per SAT
            // iteration. Prevents combinatorial blowup when many cliques are
            // detected at once; remaining cliques will be re-detected and
            // processed in subsequent iterations (smart dedup via
            // `added_resource_clique_rows` prevents double-adding).
            // Tuned higher to address combinatorial-explosion workloads like
            // infsteps180 where DDD repeatedly finds equivalent-cost solutions.
            // Adding more constraints per iter forces SAT to prove UNSAT faster.
            const MAX_CLIQUES_PER_ITER: usize = 500;
            let mut cliques_processed = 0usize;

            for (severity, members, tau, resource_a, resource_b) in clique_candidates {
                if cliques_processed >= MAX_CLIQUES_PER_ITER {
                    // Signal that more work remains so DDD loop continues.
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
                if trace_bound_queries {
                    let member_trace: Vec<(usize, usize, i32, i32)> = members
                        .iter()
                        .map(|m| (usize::from(m.visit_id), m.train_idx, m.start, m.end))
                        .collect();
                    eprintln!(
                        "[SAT-BOUND-TRACE] instance={} event=resource_clique_add iter={} resources=({}, {}) tau={} severity={} members={:?} resource_rows={} fixed_prec_rows={}",
                        problem.name,
                        iteration,
                        resource_a,
                        resource_b,
                        tau,
                        severity,
                        member_trace,
                        added_resource_clique_rows.len(),
                        fixed_prec_rows.len()
                    );
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

                // Fast-path for 2-member cliques (common case in train scheduling):
                // use direct pairwise monotone clause instead of Tseitin active_i
                // aux vars + AMO. Saves 6 Tseitin clauses + 2 aux vars per clique.
                // The clause is:
                //   [!m_end_i, !m_end_j, delay_i_past_mj_end, delay_j_past_mi_end]
                // Sound because all four literals are monotone and the semantics
                // match the AMO version: if both occupations extend to their
                // current ends, at least one must depart past the other's end.
                //
                // For cliques of size ≥ 3, fall through to the O(n) sequential
                // AMO via Tseitin `active_i(tau)` aux vars (Phase 1 encoding),
                // which scales better for large cliques.
                if members.len() == 2 {
                    let mi = members[0];
                    let mj = members[1];

                    // Capture m_end literals BEFORE timepoint creation (to keep
                    // the literal values stable w.r.t. incumbent_idx shifts).
                    let m_end_lit = |visit_id: VisitId,
                                     occupations: &TiVec<VisitId, Occ<NativeLit>>|
                     -> Bool<NativeLit> {
                        let (train_idx, visit_idx) = visits[visit_id];
                        if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                            let next_id: VisitId = (usize::from(visit_id) + 1).into();
                            occupations[next_id].delays[occupations[next_id].incumbent_idx].0
                        } else {
                            true.into()
                        }
                    };

                    let m_end_i = m_end_lit(mi.visit_id, &occupations);
                    let m_end_j = m_end_lit(mj.visit_id, &occupations);

                    let delay_i = get_delay_lit_at(
                        &mut solver,
                        problem,
                        &visits,
                        &mut occupations,
                        &mut new_time_points,
                        &mut fixed_prec_rows,
                        prec,
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
                        prec,
                        mj.visit_id,
                        mi.end,
                    );

                    solver.add_clause(vec![!m_end_i, !m_end_j, delay_i, delay_j]);
                } else {
                    // Sound sequential AMO via Tseitin-encoded "active_i(tau)" aux vars.
                    // Choose tau+1 = min(member.end) so the AMO forces at least all
                    // but one member to start at or after this time. Uses monotone
                    // delay literals that capture BOTH start and end constraints
                    // (occupation overlap at tau) — no over-constraining issue of
                    // choice literals which only capture start being in range.
                    let tau_plus_1 = members.iter().map(|m| m.end).min().unwrap();

                    let mut active_lits = Vec::with_capacity(members.len());
                    for m in &members {
                        let lit = build_active_lit(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            &mut active_lit_cache,
                            prec,
                            m.visit_id,
                            tau_plus_1,
                        );
                        active_lits.push(lit);
                    }

                    add_hybrid_amo(&mut solver, &active_lits);
                }
                n_conflict_constraints += 1;
                cliques_processed += 1;
            }

            let mut new_touched = Vec::new();
            for (i, vid) in touched_intervals.into_iter().enumerate() {
                if retain_touched[i] {
                    new_touched.push(vid);
                }
            }
            touched_intervals = new_touched;

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
                let sol = extract_solution(problem, &occupations);
                let cost = problem.cost(&sol, delay_cost_type);
                if trace_bound_queries {
                    eprintln!(
                        "[SAT-BOUND-TRACE] instance={} event=feasible_solution iter={} cost={} lb={} ub={:?} bound_used={:?}",
                        problem.name,
                        iteration,
                        cost,
                        lower_bound,
                        upper_bound,
                        bound_used
                    );
                }

                if best_sol.as_ref().map(|(c, _)| cost < *c).unwrap_or(true) {
                    best_sol = Some((cost, sol.clone()));
                    if trace_bound_queries {
                        eprintln!(
                            "[SAT-BOUND-TRACE] instance={} event=incumbent_update iter={} new_cost={} lb={} ub={:?}",
                            problem.name,
                            iteration,
                            cost,
                            lower_bound,
                            upper_bound
                        );
                    }
                    if logged_incumbent != Some(cost) {
                        value_trace.incumbent(
                            start_time,
                            cost,
                            lower_bound,
                            Some(iteration as i32),
                            Some("sat_solution"),
                        );
                        logged_incumbent = Some(cost);
                    }
                }

                if search == SatSearchMode::UbSearch {
                    let candidate_ub = cost - 1;
                    upper_bound = Some(
                        upper_bound
                            .map(|b| b.min(candidate_ub))
                            .unwrap_or(candidate_ub),
                    );
                    if use_cont_fixed_query {
                        cont_active_query_bound = None;
                    }
                }

                debug_out(DebugInfo {
                    iteration,
                    actions: Vec::new(),
                    solution: sol,
                });

                if search == SatSearchMode::UbSearch {
                    if let Some(ub) = upper_bound {
                        if ub < lower_bound {
                            let (c, s) = best_sol.clone().unwrap();
                            stats.satsolver = format!("{:?}", solver);
                            value_trace.optimal(start_time, c, Some(iteration as i32));
                            value_trace.emit(&mut output_stats, Some(c));
                            do_output_stats(
                                &mut output_stats,
                                iteration,
                                &iteration_types,
                                &stats,
                                &occupations,
                                start_time,
                                solver_time,
                                c,
                                c,
                            );
                            println!("SAT OPTIMAL (cost={})", c);
                            return Ok((s, stats));
                        }
                    }
                } else if search == SatSearchMode::Invalid {
                    invalid_clause.clear();
                    for occ in occupations.iter() {
                        let idx = occ.incumbent_idx;
                        let (lit_at, _) = occ.delays[idx];
                        invalid_clause.push(!lit_at);
                        if idx + 1 < occ.delays.len() {
                            let (lit_next, _) = occ.delays[idx + 1];
                            invalid_clause.push(lit_next);
                        }
                    }
                    if !invalid_clause.is_empty() {
                        SatInstance::add_clause(&mut solver, invalid_clause.clone());
                    }
                }
            }
        }

        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];
            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {
                match encoding {
                    SatObjectiveEncoding::Scpb => {
                        occupations[visit].cost_tree.add_cost(
                            &mut solver,
                            new_timepoint_var,
                            new_timepoint_cost,
                            &mut |weight, cost_var| {
                                let lit = cost_var
                                    .lit()
                                    .expect("CostTree produced a non-literal SCPB term");
                                scpb_terms.push((lit, weight));
                                scpb_total_weight = scpb_total_weight.saturating_add(weight);
                            },
                        );
                    }
                    SatObjectiveEncoding::IncrementalTotalizer => {
                        occupations[visit].cost_tree.add_cost(
                            &mut solver,
                            new_timepoint_var,
                            new_timepoint_cost,
                            &mut |weight, cost_var| {
                                let lit = cost_var
                                    .lit()
                                    .expect("CostTree produced a non-literal objective term");
                                budget_gte.extend([(lit, weight)]);
                            },
                        );
                    }
                    SatObjectiveEncoding::BitTotalizer => {
                        occupations[visit].cost_tree.add_cost(
                            &mut solver,
                            new_timepoint_var,
                            new_timepoint_cost,
                            &mut |weight, cost_var| {
                                let lit = cost_var
                                    .lit()
                                    .expect("CostTree produced a non-literal bit-totalizer term");
                                bit_totalizer.add_term(lit, weight);
                            },
                        );
                    }
                }
            }
        }

        if search == SatSearchMode::UbSearch {
            if let Some(ub) = upper_bound {
                if ub < lower_bound {
                    if let Some((c, s)) = best_sol.clone() {
                        stats.n_unsat += 1;
                        stats.satsolver = format!("{:?}", solver);
                        value_trace.optimal(start_time, c, Some(iteration as i32));
                        value_trace.emit(&mut output_stats, Some(c));
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            c,
                            c,
                        );
                        return Ok((s, stats));
                    }
                    return Err(SolverError::NoSolution);
                }

                let target_ub = if use_cont_fixed_query {
                    let selected = cont_active_query_bound.unwrap_or_else(|| {
                        let span = ub - lower_bound;
                        lower_bound + (span / 2)
                    });
                    let clipped = selected.min(ub).max(lower_bound);
                    cont_active_query_bound = Some(clipped);
                    clipped
                } else {
                    match mode {
                        SatBoundMode::AddClauses => ub,
                        SatBoundMode::Assumptions => (lower_bound + ub) / 2,
                    }
                };
                let ub_usize = target_ub as usize;
                match encoding {
                    SatObjectiveEncoding::Scpb => {
                        if ub_usize < scpb_total_weight {
                            bound_used = Some(target_ub);
                            match mode {
                                SatBoundMode::AddClauses => {
                                    let encoded_terms =
                                        scpb_addclauses_bounds.get(&ub_usize).copied().unwrap_or(0);
                                    if encoded_terms < scpb_terms.len() {
                                        encode_scpb_leq(&mut solver, &scpb_terms, ub_usize, None);
                                        scpb_addclauses_bounds.insert(ub_usize, scpb_terms.len());
                                    }
                                }
                                SatBoundMode::Assumptions => {
                                    let selector = match scpb_assumption_bounds.get(&ub_usize) {
                                        Some((encoded_terms, selector))
                                            if *encoded_terms == scpb_terms.len() =>
                                        {
                                            *selector
                                        }
                                        _ => {
                                            let selector = SatInstance::new_var(&mut solver);
                                            encode_scpb_leq(
                                                &mut solver,
                                                &scpb_terms,
                                                ub_usize,
                                                Some(selector),
                                            );
                                            scpb_assumption_bounds
                                                .insert(ub_usize, (scpb_terms.len(), selector));
                                            selector
                                        }
                                    };
                                    bound_assumptions.push(selector);
                                }
                            }
                        }
                    }
                    SatObjectiveEncoding::BitTotalizer => {
                        if ub_usize < bit_totalizer.total_weight {
                            bound_used = Some(target_ub);
                            let term_count = bit_totalizer.term_count();
                            match mode {
                                SatBoundMode::AddClauses => {
                                    let encoded_terms = bit_totalizer
                                        .addclauses_bounds
                                        .get(&ub_usize)
                                        .copied()
                                        .unwrap_or(0);
                                    if encoded_terms < term_count {
                                        bit_totalizer.encode_leq(&mut solver, ub_usize, None);
                                        bit_totalizer
                                            .addclauses_bounds
                                            .insert(ub_usize, term_count);
                                    }
                                }
                                SatBoundMode::Assumptions => {
                                    let cached =
                                        bit_totalizer.assumption_bounds.get(&ub_usize).copied();
                                    let selector = match cached {
                                        Some((encoded_terms, selector))
                                            if encoded_terms == term_count =>
                                        {
                                            selector
                                        }
                                        _ => {
                                            let selector = SatInstance::new_var(&mut solver);
                                            bit_totalizer.encode_leq(
                                                &mut solver,
                                                ub_usize,
                                                Some(selector),
                                            );
                                            bit_totalizer
                                                .assumption_bounds
                                                .insert(ub_usize, (term_count, selector));
                                            selector
                                        }
                                    };
                                    bound_assumptions.push(selector);
                                }
                            }
                        }
                    }
                    SatObjectiveEncoding::IncrementalTotalizer => {
                        if ub_usize < budget_gte.weight_sum() {
                            let bound_lits: Vec<Bool<NativeLit>> = {
                                let (inner, next_var) = (&mut solver.inner, &mut solver.next_var);
                                let mut collector = NativeClauseCollector { inner };
                                let mut var_manager = NativeVarManager { next_var };
                                budget_gte
                                    .encode_ub_change(
                                        0..=ub_usize,
                                        &mut collector,
                                        &mut var_manager,
                                    )
                                    .map_err(|_| SolverError::OutOfMemory)?;

                                budget_gte
                                    .enforce_ub(ub_usize)
                                    .unwrap_or_else(|err| {
                                        panic!(
                                            "failed to enforce GeneralizedTotalizer upper bound {}: {:?}",
                                            ub_usize, err
                                        )
                                    })
                                    .into_iter()
                                    .map(Bool::from_lit)
                                    .collect()
                            };

                            if !bound_lits.is_empty() {
                                bound_used = Some(target_ub);
                                match mode {
                                    SatBoundMode::AddClauses => {
                                        if last_added_bound != Some(ub_usize) {
                                            for lit in bound_lits {
                                                SatInstance::add_clause(&mut solver, vec![lit]);
                                            }
                                            last_added_bound = Some(ub_usize);
                                        }
                                    }
                                    SatBoundMode::Assumptions => {
                                        bound_assumptions.extend(bound_lits);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        *iteration_types.entry(IterationType::Objective).or_default() += 1;

        let solver_debug = format!("{:?}", solver);
        let remaining_timeout = (timeout - start_time.elapsed().as_secs_f64()).max(0.0);
        solver.set_solve_timeout(Some(Duration::from_secs_f64(remaining_timeout)));
        let bound_assumption_count = bound_assumptions.len();
        if trace_bound_queries {
            let total_delay_points: usize = occupations.iter().map(|occ| occ.delays.len()).sum();
            eprintln!(
                "[SAT-BOUND-TRACE] instance={} event=query iter={} search={:?} mode={:?} encoding={:?} lb={} ub={:?} bound_used={:?} best_sol={:?} assumptions={} delay_points={} touched={} fixed_prec_rows={} resource_rows={}",
                problem.name,
                iteration,
                search,
                mode,
                encoding,
                lower_bound,
                upper_bound,
                bound_used,
                best_sol.as_ref().map(|(c, _)| *c),
                bound_assumption_count,
                total_delay_points,
                touched_intervals.len(),
                fixed_prec_rows.len(),
                added_resource_clique_rows.len()
            );
        }
        println!(
            "SAT iteration {} LB={} UB={:?} query_bound={:?}",
            iteration, lower_bound, upper_bound, bound_used
        );
        let solve_start = Instant::now();
        let result = solver.solve_with_assumptions_owned(bound_assumptions.into_iter());
        solver_time += solve_start.elapsed();

        match result {
            NativeSolveResult::Sat(model) => {
                if trace_bound_queries {
                    eprintln!(
                        "[SAT-BOUND-TRACE] instance={} event=result iter={} result=SAT lb={} ub={:?} bound_used={:?} best_sol_before={:?}",
                        problem.name,
                        iteration,
                        lower_bound,
                        upper_bound,
                        bound_used,
                        best_sol.as_ref().map(|(c, _)| *c)
                    );
                }
                is_sat = true;
                stats.n_sat += 1;

                let mut touched_seen = vec![false; occupations.len()];
                if !touched_intervals.is_empty() {
                    let mut write = 0usize;
                    for read in 0..touched_intervals.len() {
                        let vid = touched_intervals[read];
                        let idx = usize::from(vid);
                        if !touched_seen[idx] {
                            touched_seen[idx] = true;
                            touched_intervals[write] = vid;
                            write += 1;
                        }
                    }
                    touched_intervals.truncate(write);
                }

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

                    let (_, visit_idx) = visits[visit];
                    if touched {
                        if visit_idx > 0 {
                            let prev_visit = (Into::<usize>::into(visit) - 1).into();
                            let prev_idx = usize::from(prev_visit);
                            if !touched_seen[prev_idx] {
                                touched_seen[prev_idx] = true;
                                touched_intervals.push(prev_visit);
                            }
                        }
                        let visit_u = usize::from(visit);
                        if !touched_seen[visit_u] {
                            touched_seen[visit_u] = true;
                            touched_intervals.push(visit);
                        }
                    }
                }
            }
            NativeSolveResult::Unsat(_core) => {
                if trace_bound_queries {
                    eprintln!(
                        "[SAT-BOUND-TRACE] instance={} event=result iter={} result=UNSAT lb_before={} ub={:?} bound_used={:?} best_sol={:?}",
                        problem.name,
                        iteration,
                        lower_bound,
                        upper_bound,
                        bound_used,
                        best_sol.as_ref().map(|(c, _)| *c)
                    );
                }
                is_sat = false;
                stats.n_unsat += 1;
                if search == SatSearchMode::Invalid {
                    if let Some((c, s)) = best_sol.clone() {
                        stats.satsolver = solver_debug;
                        value_trace.optimal(start_time, c, Some(iteration as i32));
                        value_trace.emit(&mut output_stats, Some(c));
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            c,
                            c,
                        );
                        return Ok((s, stats));
                    }
                    return Err(SolverError::NoSolution);
                }

                if let Some(bound) = bound_used {
                    lower_bound = bound + 1;
                    if trace_bound_queries {
                        eprintln!(
                            "[SAT-BOUND-TRACE] instance={} event=lower_bound_update iter={} from_unsat_bound={} new_lb={} ub={:?} best_sol={:?}",
                            problem.name,
                            iteration,
                            bound,
                            lower_bound,
                            upper_bound,
                            best_sol.as_ref().map(|(c, _)| *c)
                        );
                    }
                    if logged_lower_bound != Some(lower_bound) {
                        value_trace.lower_bound(
                            start_time,
                            lower_bound,
                            best_sol.as_ref().map(|(c, _)| *c),
                            Some(iteration as i32),
                            Some("unsat_bound"),
                        );
                        logged_lower_bound = Some(lower_bound);
                    }
                    if use_cont_fixed_query {
                        cont_active_query_bound = None;
                    }
                    if let (Some((c, s)), Some(ub)) = (best_sol.clone(), upper_bound) {
                        if ub < lower_bound {
                            stats.satsolver = solver_debug;
                            value_trace.optimal(start_time, c, Some(iteration as i32));
                            value_trace.emit(&mut output_stats, Some(c));
                            do_output_stats(
                                &mut output_stats,
                                iteration,
                                &iteration_types,
                                &stats,
                                &occupations,
                                start_time,
                                solver_time,
                                c,
                                c,
                            );
                            println!("SAT OPTIMAL (cost={})", c);
                            return Ok((s, stats));
                        }
                    }
                    iteration += 1;
                    continue;
                }

                return Err(SolverError::NoSolution);
            }
            NativeSolveResult::Interrupted => {
                let ub = best_sol.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX);
                let lb = lower_bound;
                println!("TIMEOUT LB={} UB={}", lb, ub);
                value_trace.timeout(start_time, lb, best_sol.as_ref().map(|(c, _)| *c), Some(iteration as i32));
                value_trace.emit(&mut output_stats, best_sol.as_ref().map(|(c, _)| *c));
                do_output_stats(
                    &mut output_stats,
                    iteration,
                    &iteration_types,
                    &stats,
                    &occupations,
                    start_time,
                    solver_time,
                    lb,
                    ub,
                );
                return Err(SolverError::Timeout);
            }
        }

        iteration += 1;
    }
}

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
    prec: SatPrecEncoding,
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
    match prec {
        SatPrecEncoding::Plain => {
            solver.add_clause(vec![!in_var, req_var]);
        }
        SatPrecEncoding::Scl => {
            const SCL_PAIRWISE_THRESHOLD: usize = 5;
            let idx = occupations[next_visit]
                .delays
                .partition_point(|(_, t0)| *t0 < req_t);
            if idx <= SCL_PAIRWISE_THRESHOLD {
                for i in 0..idx {
                    let lit_i = occupations[next_visit].delays[i].0;
                    let lit_next = occupations[next_visit].delays[i + 1].0;
                    solver.add_clause(vec![!in_var, !lit_i, lit_next]);
                }
            } else {
                solver.add_clause(vec![!in_var, req_var]);
            }
        }
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
    prec: SatPrecEncoding,
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
            prec,
        ) {
            queue.push_back(next);
        }
    }
}
