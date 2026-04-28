pub mod maxsatddd_ladder;
pub mod bigm;
pub mod binarizedbigm;
pub mod greedy;
pub mod idl;
pub mod mipdddpack;
mod minimize;
pub mod maxsat_ti;
pub mod maxsat_ddd;
pub mod maxsatddd_ladder_scl;
pub mod heuristic;
pub mod maxsatddd_ladder_abstract;
pub mod milp_ti;
pub mod ddd;
pub mod value_trace;
// pub mod cutting;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
    OutOfMemory,
}
