use std::time::Instant;

use serde_json::{json, Map, Value};

#[derive(Default)]
pub struct ValueTrace {
    events: Vec<TraceEvent>,
}

struct TraceEvent {
    event: &'static str,
    elapsed_ms: f64,
    iteration: Option<i32>,
    value: Option<i32>,
    incumbent: Option<i32>,
    lower_bound: Option<i32>,
    note: Option<&'static str>,
}

impl TraceEvent {
    fn to_json(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("event".to_string(), self.event.into());
        obj.insert("elapsed_ms".to_string(), self.elapsed_ms.into());
        if let Some(iteration) = self.iteration {
            obj.insert("iteration".to_string(), iteration.into());
        }
        if let Some(value) = self.value {
            obj.insert("value".to_string(), value.into());
        }
        if let Some(incumbent) = self.incumbent {
            obj.insert("incumbent".to_string(), incumbent.into());
        }
        if let Some(lower_bound) = self.lower_bound {
            obj.insert("lower_bound".to_string(), lower_bound.into());
        }
        if let Some(note) = self.note {
            obj.insert("note".to_string(), note.into());
        }
        Value::Object(obj)
    }
}

impl ValueTrace {
    fn push(
        &mut self,
        start_time: Instant,
        event: &'static str,
        iteration: Option<i32>,
        value: Option<i32>,
        incumbent: Option<i32>,
        lower_bound: Option<i32>,
        note: Option<&'static str>,
    ) {
        self.events.push(TraceEvent {
            event,
            elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            iteration,
            value,
            incumbent,
            lower_bound,
            note,
        });
    }

    pub fn initial_incumbent(
        &mut self,
        start_time: Instant,
        incumbent: i32,
        lower_bound: i32,
        iteration: Option<i32>,
    ) {
        self.push(
            start_time,
            "initial_incumbent",
            iteration,
            Some(incumbent),
            Some(incumbent),
            Some(lower_bound),
            None,
        );
    }

    pub fn incumbent(
        &mut self,
        start_time: Instant,
        incumbent: i32,
        lower_bound: i32,
        iteration: Option<i32>,
        note: Option<&'static str>,
    ) {
        self.push(
            start_time,
            "incumbent",
            iteration,
            Some(incumbent),
            Some(incumbent),
            Some(lower_bound),
            note,
        );
    }

    pub fn lower_bound(
        &mut self,
        start_time: Instant,
        lower_bound: i32,
        incumbent: Option<i32>,
        iteration: Option<i32>,
        note: Option<&'static str>,
    ) {
        self.push(
            start_time,
            "lower_bound",
            iteration,
            lower_bound.checked_sub(1),
            incumbent,
            Some(lower_bound),
            note,
        );
    }

    pub fn optimal(&mut self, start_time: Instant, value: i32, iteration: Option<i32>) {
        self.push(
            start_time,
            "optimal",
            iteration,
            Some(value),
            Some(value),
            Some(value),
            None,
        );
    }

    pub fn timeout(
        &mut self,
        start_time: Instant,
        lower_bound: i32,
        incumbent: Option<i32>,
        iteration: Option<i32>,
    ) {
        self.push(
            start_time,
            "timeout",
            iteration,
            None,
            incumbent,
            Some(lower_bound),
            None,
        );
    }

    pub fn emit(
        &self,
        output_stats: &mut impl FnMut(String, Value),
        final_cost: Option<i32>,
    ) {
        output_stats(
            "value_trace_version".to_string(),
            json!(1),
        );
        output_stats(
            "value_trace".to_string(),
            Value::Array(self.events.iter().map(TraceEvent::to_json).collect()),
        );

        let first_solution_ms = self
            .events
            .iter()
            .find(|event| event.incumbent.is_some())
            .map(|event| event.elapsed_ms);
        if let Some(first_solution_ms) = first_solution_ms {
            output_stats(
                "time_to_first_solution_ms".to_string(),
                first_solution_ms.into(),
            );
        }

        if let Some(final_cost) = final_cost {
            let time_to_best_value_ms = self
                .events
                .iter()
                .find(|event| event.incumbent == Some(final_cost))
                .map(|event| event.elapsed_ms);
            if let Some(time_to_best_value_ms) = time_to_best_value_ms {
                output_stats(
                    "time_to_best_value_ms".to_string(),
                    time_to_best_value_ms.into(),
                );
            }

            let time_to_prove_best_value_ms = self
                .events
                .iter()
                .find(|event| event.lower_bound.is_some_and(|lb| lb >= final_cost))
                .map(|event| event.elapsed_ms);
            if let Some(time_to_prove_best_value_ms) = time_to_prove_best_value_ms {
                output_stats(
                    "time_to_prove_best_value_ms".to_string(),
                    time_to_prove_best_value_ms.into(),
                );
            }

            if let (Some(best_ms), Some(prove_ms)) =
                (time_to_best_value_ms, time_to_prove_best_value_ms)
            {
                output_stats(
                    "time_to_optimality_ms".to_string(),
                    best_ms.max(prove_ms).into(),
                );
            }
        }
    }
}
