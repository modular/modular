use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
pub struct RustMetrics {
    requests_started: AtomicU64,
    requests_completed: AtomicU64,
    requests_cancelled: AtomicU64,
    requests_rejected: AtomicU64,
    first_token_count: AtomicU64,
    first_token_latency_us_total: AtomicU64,
    end_to_end_count: AtomicU64,
    end_to_end_latency_us_total: AtomicU64,
    decode_calls: AtomicU64,
    decode_tokens_total: AtomicU64,
    decode_latency_us_total: AtomicU64,
    request_batches: AtomicU64,
    request_batch_items_total: AtomicU64,
}

#[derive(Debug, Serialize)]
pub struct RustMetricsSnapshot {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_cancelled: u64,
    pub requests_rejected: u64,
    pub first_token_count: u64,
    pub first_token_latency_us_total: u64,
    pub first_token_latency_us_avg: f64,
    pub end_to_end_count: u64,
    pub end_to_end_latency_us_total: u64,
    pub end_to_end_latency_us_avg: f64,
    pub decode_calls: u64,
    pub decode_tokens_total: u64,
    pub decode_latency_us_total: u64,
    pub decode_latency_us_avg: f64,
    pub request_batches: u64,
    pub request_batch_items_total: u64,
    pub request_batch_items_avg: f64,
}

impl RustMetrics {
    pub fn record_request_started(&self) {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_completed(&self) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_cancelled(&self) {
        self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_rejected(&self) {
        self.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_first_token_latency(&self, duration: Duration) {
        self.first_token_count.fetch_add(1, Ordering::Relaxed);
        self.first_token_latency_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_end_to_end_latency(&self, duration: Duration) {
        self.end_to_end_count.fetch_add(1, Ordering::Relaxed);
        self.end_to_end_latency_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_decode(&self, tokens: usize, duration: Duration) {
        self.decode_calls.fetch_add(1, Ordering::Relaxed);
        self.decode_tokens_total
            .fetch_add(tokens as u64, Ordering::Relaxed);
        self.decode_latency_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_request_batch(&self, batch_size: usize) {
        self.request_batches.fetch_add(1, Ordering::Relaxed);
        self.request_batch_items_total
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> RustMetricsSnapshot {
        let requests_started = self.requests_started.load(Ordering::Relaxed);
        let requests_completed = self.requests_completed.load(Ordering::Relaxed);
        let requests_cancelled = self.requests_cancelled.load(Ordering::Relaxed);
        let requests_rejected = self.requests_rejected.load(Ordering::Relaxed);
        let first_token_count = self.first_token_count.load(Ordering::Relaxed);
        let first_token_latency_us_total =
            self.first_token_latency_us_total.load(Ordering::Relaxed);
        let end_to_end_count = self.end_to_end_count.load(Ordering::Relaxed);
        let end_to_end_latency_us_total =
            self.end_to_end_latency_us_total.load(Ordering::Relaxed);
        let decode_calls = self.decode_calls.load(Ordering::Relaxed);
        let decode_tokens_total = self.decode_tokens_total.load(Ordering::Relaxed);
        let decode_latency_us_total = self.decode_latency_us_total.load(Ordering::Relaxed);
        let request_batches = self.request_batches.load(Ordering::Relaxed);
        let request_batch_items_total =
            self.request_batch_items_total.load(Ordering::Relaxed);

        RustMetricsSnapshot {
            requests_started,
            requests_completed,
            requests_cancelled,
            requests_rejected,
            first_token_count,
            first_token_latency_us_total,
            first_token_latency_us_avg: avg(first_token_latency_us_total, first_token_count),
            end_to_end_count,
            end_to_end_latency_us_total,
            end_to_end_latency_us_avg: avg(end_to_end_latency_us_total, end_to_end_count),
            decode_calls,
            decode_tokens_total,
            decode_latency_us_total,
            decode_latency_us_avg: avg(decode_latency_us_total, decode_calls),
            request_batches,
            request_batch_items_total,
            request_batch_items_avg: avg(request_batch_items_total, request_batches),
        }
    }
}

fn avg(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}
