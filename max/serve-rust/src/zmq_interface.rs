use crate::types::{RequestID, SchedulerResult, TextGenerationContext};
use futures::Stream;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use zeromq::{PushSocket, PullSocket, Socket, SocketRecv, SocketSend};
use serde::{Deserialize, Serialize};

pub struct ZmqModelWorkerProxy<Request, Reply> {
    request_push: Arc<Mutex<PushSocket>>,
    response_pull: Arc<Mutex<PullSocket>>,
    cancel_push: Arc<Mutex<PushSocket>>,
    pending_out_queues: Arc<Mutex<HashMap<RequestID, mpsc::Sender<SchedulerResult<Reply>>>>>,
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
    ) -> Self {
        let mut request_push = PushSocket::new();
        request_push.bind(request_addr).await.expect("Failed to bind request socket");

        let mut response_pull = PullSocket::new();
        response_pull.bind(response_addr).await.expect("Failed to bind response socket");

        let mut cancel_push = PushSocket::new();
        cancel_push.bind(cancel_addr).await.expect("Failed to bind cancel socket");

        Self {
            request_push: Arc::new(Mutex::new(request_push)),
            response_pull: Arc::new(Mutex::new(response_pull)),
            cancel_push: Arc::new(Mutex::new(cancel_push)),
            pending_out_queues: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    pub async fn stream(
        &self,
        request_id: RequestID,
        data: Request,
    ) -> Pin<Box<dyn Stream<Item = Vec<Reply>> + Send>> {
        let (tx, mut rx) = mpsc::channel(100);
        {
            let mut pending = self.pending_out_queues.lock().await;
            pending.insert(request_id.clone(), tx);
        }

        // Send request with RequestID for context
        let context = TextGenerationContext {
            request_id: request_id.clone(),
            request: data,
        };

        let mut push = self.request_push.lock().await;
        let serialized = rmp_serde::to_vec(&context).expect("Serialization failed");
        push.send(serialized.into()).await.expect("Send failed");

        let stream = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                if let Some(result) = item.result {
                    yield vec![result];
                }
                if item.is_done {
                    break;
                }
            }
        };

        Box::pin(stream)
    }

    pub async fn start_response_worker(self: Arc<Self>) {
        let proxy = Arc::clone(&self);
        tokio::spawn(async move {
            loop {
                let msg = {
                    let mut pull = proxy.response_pull.lock().await;
                    pull.recv().await.expect("Recv failed")
                };

                let response_dict: HashMap<RequestID, SchedulerResult<Reply>> =
                    rmp_serde::from_slice(&msg.get(0).unwrap()).expect("Deserialization failed");

                let mut pending = proxy.pending_out_queues.lock().await;
                for (request_id, response) in response_dict {
                    if let Some(tx) = pending.get(&request_id) {
                        if tx.send(response).await.is_err() {
                            // Handler dropped, clean up
                        }
                    }
                }
            }
        });
    }
}
