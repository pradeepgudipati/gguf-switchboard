//! TypeSafe System One API for decision models (`POST /v1/systemone`).
//!
//! Decision models (e.g. Bespoke-Nimble) don't generate text: they read a
//! `state` once and score the typed `questions` you supply (`choice`, `score`
//! or `noul`), returning a probability per option in a single forward pass.
//! The request/response envelope below mirrors llama.cpp's server, which is
//! where the request is proxied.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "bespoke-nimble-9b-v3",
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "payments, charges, refunds, invoices",
                "shipping": "delivery, tracking, lost or late parcels",
                "technical": "bugs, errors, login problems"
            }
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"}
    }
}))]
pub struct SystemoneRequest {
    /// Decision-model id as listed in `/v1/models` (`kind = "decision"`).
    pub model: String,
    /// Text, JSON object, or chat-message list the questions are scored against.
    pub state: Value,
    /// Named questions (`choice` | `score` | `noul`), each with
    /// `instructions` and optional `criteria`. Kept as free-form JSON so new
    /// question shapes pass through untouched.
    pub questions: HashMap<String, Value>,
    /// Data-URL images for vision-capable decision models (alias: `files`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    #[serde(flatten, default)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SystemoneResponse {
    pub model: String,
    /// Per-question answers (`choice` + `probabilities`, `score` + `legend`,
    /// or `noul`), each with `type` and `confidence` where applicable.
    pub answers: HashMap<String, Value>,
    pub usage: SystemoneUsage,
    #[serde(flatten, default)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SystemoneUsage {
    #[serde(default)]
    pub input_tokens: u32,
    /// Always 0 — nothing is sampled.
    #[serde(default)]
    pub output_tokens: u32,
}
