# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartKnowledgeBase:
    def __init__(
        self, endpoint: str = "http://localhost:8000/v1/embeddings"
    ) -> None:
        self.endpoint = endpoint
        self.documents: list[str] = []
        self.doc_titles: list[str] = []
        self.embeddings: np.ndarray = None
        self.clusters: dict[int, list[int]] = {}

    def _get_embedding(
        self, texts: list[str], max_retries: int = 3
    ) -> np.ndarray:
        """Get embeddings with retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers={"Content-Type": "application/json"},
                    json={
                        "input": texts,
                        "model": "sentence-transformers/all-mpnet-base-v2",
                    },
                    timeout=5,
                ).json()
                return np.array(
                    [item["embedding"] for item in response["data"]]
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(  # noqa: B904
                        f"Failed to get embeddings after {max_retries} attempts: {e}"
                    )
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")

    @lru_cache(maxsize=1000)  # noqa: B019
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Cached version for single text embedding."""
        return self._get_embedding([text])[0]

    def add_document(self, title: str, content: str) -> None:
        """Add a single document with title."""
        self.doc_titles.append(title)
        self.documents.append(content)

        # Update embeddings
        if len(self.documents) == 1:
            self.embeddings = self._get_embedding([content])
        else:
            self.embeddings = np.vstack(
                [self.embeddings, self._get_embedding([content])]
            )

        # Recluster if we have enough documents
        if len(self.documents) >= 3:
            self._cluster_documents()

    def _cluster_documents(self, n_clusters: Optional[int] = None) -> None:
        """Cluster documents into topics."""
        if n_clusters is None:
            n_clusters = max(2, len(self.documents) // 5)

        n_clusters = min(n_clusters, len(self.documents))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(
            self.embeddings
        )

        self.clusters = {}
        for i in range(n_clusters):
            self.clusters[i] = np.where(kmeans.labels_ == i)[0].tolist()

    def search(
        self, query: str, top_k: int = 3
    ) -> list[tuple[str, str, float]]:
        """Find documents most similar to the query."""
        query_embedding = self._get_embedding_cached(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            (self.doc_titles[i], self.documents[i], similarities[i])
            for i in top_indices
        ]

    def get_topic_documents(self, topic_id: int) -> list[tuple[str, str]]:
        """Get all documents in a topic cluster."""
        return [
            (self.doc_titles[i], self.documents[i])
            for i in self.clusters.get(topic_id, [])
        ]

    def suggest_topics(
        self, query: str, top_k: int = 2
    ) -> list[tuple[int, float]]:
        query_embedding = self._get_embedding_cached(query)
        topic_similarities = []

        for topic_id, doc_indices in self.clusters.items():
            topic_embeddings = self.embeddings[doc_indices]
            similarity = cosine_similarity(
                [query_embedding], topic_embeddings
            ).max()
            topic_similarities.append((topic_id, similarity))  # Remove [0]

        return sorted(topic_similarities, key=lambda x: x[1], reverse=True)[
            :top_k
        ]


# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = SmartKnowledgeBase()

    # Add technical documentation
    kb.add_document(
        "Password Reset Guide",
        "To reset your password: 1. Click 'Forgot Password' 2. Enter your email "
        "3. Follow the reset link 4. Create a new password meeting security requirements",
    )

    kb.add_document(
        "Account Security",
        "Secure your account by enabling 2FA, using a strong password, and regularly "
        "monitoring account activity. Enable login notifications for suspicious activity.",
    )

    kb.add_document(
        "Billing Overview",
        "Your billing cycle starts on the 1st of each month. View charges, update "
        "payment methods, and download invoices from the Billing Dashboard.",
    )

    kb.add_document(
        "Payment Methods",
        "We accept credit cards, PayPal, and bank transfers. Update payment methods "
        "in Billing Settings. New payment methods are verified with a $1 hold.",
    )

    kb.add_document(
        "Installation Guide",
        "Install by downloading the appropriate package for your OS. Run with admin "
        "privileges. Follow prompts to select installation directory and components.",
    )

    kb.add_document(
        "System Requirements",
        "Minimum: 8GB RAM, 2GB storage, Windows 10/macOS 11+. Recommended: 16GB RAM, "
        "4GB storage, SSD, modern multi-core processor for optimal performance.",
    )

    # Example 1: Search for password-related help
    print("\nSearching for password help:")
    results = kb.search("How do I change my password?")
    for title, content, score in results:
        print(f"\nTitle: {title}")
        print(f"Relevance: {score:.2f}")
        print(f"Content: {content[:100]}...")

    # Example 2: Get topic suggestions
    print("\nGetting topics for billing query:")
    query = "Where can I update my credit card?"
    topics = kb.suggest_topics(query)
    for topic_id, relevance in topics:
        print(f"\nTopic {topic_id} (Relevance: {relevance:.2f}):")
        for title, content in kb.get_topic_documents(topic_id):
            print(f"- {title}: {content[:50]}...")

    # Example 3: Get all documents in a topic
    print("\nAll documents in Topic 0:")
    for title, content in kb.get_topic_documents(0):
        print(f"\nTitle: {title}")
        print(f"Content: {content[:100]}...")
