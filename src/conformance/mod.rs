//! Tool-calling / template conformance console.
//!
//! This module backs a debug surface (not a chat UI) for diagnosing why a
//! local model breaks OpenAI-compatible tool calling or ships a broken
//! embedded Jinja chat template — the two failure modes that block agentic
//! harnesses (opencode, etc.) from using a model reliably. It intentionally
//! reuses the existing request/response types and the `Backend` trait rather
//! than inventing a parallel pipeline.

pub mod battery;
pub mod classify;
pub mod history;

pub use history::{ConformanceHistory, ConformanceRunDetail, ConformanceRunSummary};
