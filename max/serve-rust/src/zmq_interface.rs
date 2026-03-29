use crate::types::{RequestID, SchedulerResult, TextGenerationContext};
use crate::metrics::RustMetrics;
use futures::Stream;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::{mpsc, Mutex};
use zeromq::{PushSocket, PullSocket, Socket, SocketRecv, SocketSend};
use serde::{Deserialize, Serialize};

pub struct ZmqProxyConfig {
    pub request_queue_capacity: usize,
    pub cancel_queue_capacity: usize,
    pub request_batch_max_size: usize,
    pub request_batch_wait: Duration,
}

impl Default for ZmqProxyConfig {
    fn default() -> Self {
        Self {
            request_queue_capacity: 4096,
            cancel_queue_capacity: 1024,
            request_batch_max_size: 32,
            request_batch_wait: Duration::from_micros(200),
        }
    }
}

struct OutboundRequest {
    payload: Vec<u8>,
}

struct PendingRequest<Reply> {
    tx: mpsc::UnboundedSender<SchedulerResult<Reply>>,
    started_at: Instant,
    seen_first_token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamInitError {
    Overloaded,
    Unavailable,
}

impl fmt::Display for StreamInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamInitError::Overloaded => write!(f, "request queue is overloaded"),
            StreamInitError::Unavailable => write!(f, "request queue is unavailable"),
        }
    }
}

pub struct ZmqModelWorkerProxy<Request, Reply> {
    response_pull: Arc<Mutex<PullSocket>>,
    request_tx: mpsc::Sender<OutboundRequest>,
    cancel_tx: mpsc::Sender<Vec<u8>>,
    metrics: Arc<RustMetrics>,
    pending_out_queues:
        Arc<Mutex<HashMap<RequestID, PendingRequest<Reply>>>>,
    _phantom: PhantomData<Request>,
}

impl<Request, Reply> ZmqModelWorkerProxy<Request, Reply>
where
    Request: Serialize + Send + Sync + 'static,
    Reply: for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    pub async fn new(
        request_addr: &str,
        response_addr: &str,
        cancel_addr: &str,
        cfg: ZmqProxyConfig,
        metrics: Arc<RustMetrics>,
    ) -> Self {
        let mut request_push = PushSocket::new();
        request_push.bind(request_addr).await.expect("Failed to bind request socket");

        let mut response_pull = PullSocket::new();
        response_pull.bind(response_addr).await.expect("Failed to bind response socket");

        let mut cancel_push = PushSocket::new();
        cancel_push.bind(cancel_addr).await.expect("Failed to bind cancel socket");

        let (request_tx, mut request_rx) =
            mpsc::channel::<OutboundRequest>(cfg.request_queue_capacity);
        let sender_metrics = Arc::clone(&metrics);
        let max_batch_size = cfg.request_batch_max_size.max(1);
        let batch_wait = cfg.request_batch_wait;
        tokio::spawn(async move {
            while let Some(first) = request_rx.recv().await {
                let mut batch = vec![first];
                let deadline = Instant::now() + batch_wait;

                while batch.len() < max_batch_size {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }

                    match tokio::time::timeout(remaining, request_rx.recv()).await {
                        Ok(Some(msg)) => batch.push(msg),
                        Ok(None) | Err(_) => break,
                    }
                }

                sender_metrics.record_request_batch(batch.len());
                for msg in batch {
                    if let Err(err) = request_push.send(msg.payload.into()).await {
                        tracing::error!("Failed to send request message over ZMQ: {}", err);
                    }
                }
            }
        });

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<Vec<u8>>(cfg.cancel_queue_capacity);
        tokio::spawn(async move {
            while let Some(msg) = cancel_rx.recv().await {
                if let Err(err) = cancel_push.send(msg.into()).await {
                    tracing::error!("Failed to send cancellation message over ZMQ: {}", err);
                }
            }
        });

        Self {
            response_pull: Arc::new(Mutex::new(response_pull)),
            request_tx,
            cancel_tx,
            metrics,
            pending_out_queues: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    pub async fn stream(
        &self,
        request_id: RequestID,
        data: Request,
    ) -> Result<Pin<Box<dyn Stream<Item = Vec<Reply>> + Send>>, StreamInitError> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        {
            let mut pending = self.pending_out_queues.lock().await;
            pending.insert(
                request_id.clone(),
                PendingRequest {
                    tx,
                    started_at: Instant::now(),
                    seen_first_token: false,
                },
            );
        }

        // Send request with RequestID for context
        let context = TextGenerationContext {
            request_id: request_id.clone(),
            request: data,
        };

        let serialized = rmp_serde::to_vec(&context).expect("Serialization failed");
        match self
            .request_tx
            .try_send(OutboundRequest { payload: serialized })
        {
            Ok(()) => {
                self.metrics.record_request_started();
            }
            Err(TrySendError::Full(_)) => {
                let mut pending = self.pending_out_queues.lock().await;
                pending.remove(&request_id);
                self.metrics.record_request_rejected();
                return Err(StreamInitError::Overloaded);
            }
            Err(TrySendError::Closed(_)) => {
                let mut pending = self.pending_out_queues.lock().await;
                pending.remove(&request_id);
                self.metrics.record_request_rejected();
                return Err(StreamInitError::Unavailable);
            }
        }

        let stream = async_stream::stream! {
            while let Some(first_item) = rx.recv().await {
                let mut outputs = Vec::new();
                let mut should_stop = false;

                if let Some(result) = first_item.result {
                    outputs.push(result);
                }
                if first_item.is_done {
                    should_stop = true;
                }

                while !should_stop {
                    match rx.try_recv() {
                        Ok(item) => {
                            if let Some(result) = item.result {
                                outputs.push(result);
                            }
                            if item.is_done {
                                should_stop = true;
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            should_stop = true;
                            break;
                        }
                    }
                }

                if !outputs.is_empty() {
                    yield outputs;
                }

                if should_stop {
                    break;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    pub fn start_response_worker(self: Arc<Self>) {
        let proxy = Arc::clone(&self);
        tokio::spawn(async move {
            loop {
                let msg = {
                    let mut pull = proxy.response_pull.lock().await;
                    pull.recv().await.expect("Recv failed")
                };

                let response_part = match msg.get(0) {
                    Some(part) => part,
                    None => {
                        tracing::error!("Received empty ZMQ message");
                        continue;
                    }
                };
                let response_dict: HashMap<RequestID, SchedulerResult<Reply>> =
                    rmp_serde::from_slice(response_part).expect("Deserialization failed");

                let mut deliveries = Vec::with_capacity(response_dict.len());
                let mut to_remove = Vec::new();
                let mut to_cancel = Vec::new();
                let mut completed_count = 0_u64;
                let mut cancelled_count = 0_u64;

                {
                    let mut pending = proxy.pending_out_queues.lock().await;
                    for (request_id, response) in response_dict {
                        if let Some(entry) = pending.get_mut(&request_id) {
                            if !entry.seen_first_token && response.result.is_some() {
                                entry.seen_first_token = true;
                                proxy
                                    .metrics
                                    .record_first_token_latency(entry.started_at.elapsed());
                            }
                            deliveries.push((
                                request_id,
                                entry.tx.clone(),
                                entry.started_at,
                                response,
                            ));
                        }
                    }
                }

                for (request_id, tx, started_at, response) in deliveries {
                    let is_done = response.is_done;
                    if tx.send(response).is_err() {
                        // Client disconnected, mark for removal and cancellation.
                        to_remove.push(request_id.clone());
                        to_cancel.push(request_id);
                        cancelled_count += 1;
                    } else if is_done {
                        // Request finished, mark for removal.
                        proxy
                            .metrics
                            .record_end_to_end_latency(started_at.elapsed());
                        completed_count += 1;
                        to_remove.push(request_id);
                    }
                }

                let mut pending = proxy.pending_out_queues.lock().await;
                for id in &to_remove {
                    pending.remove(id);
                }

                for _ in 0..completed_count {
                    proxy.metrics.record_request_completed();
                }
                for _ in 0..cancelled_count {
                    proxy.metrics.record_request_cancelled();
                }

                if !to_cancel.is_empty() {
                    // Unlock pending queues before enqueueing cancellation to avoid potential deadlock.
                    drop(pending);
                    let serialized = rmp_serde::to_vec(&to_cancel).expect("Serialization failed");
                    if let Err(e) = proxy.cancel_tx.send(serialized).await {
                        tracing::error!("Failed to enqueue cancellation message: {}", e);
                    }
                }
            }
        });
    }

    pub fn metrics_snapshot(&self) -> crate::metrics::RustMetricsSnapshot {
        self.metrics.snapshot()
    }
}
