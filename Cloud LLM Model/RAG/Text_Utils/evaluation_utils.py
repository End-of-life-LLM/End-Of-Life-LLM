"""Utilities for evaluating RAG system performance."""

from typing import Any, Callable, Dict, List


class Evaluation_Utils:
    """Class for evaluating retrieval and RAG system performance."""
    
    @staticmethod
    def evaluate_retrieval(retriever, 
                          embedding_function: Callable[[str], List[float]], 
                          questions: List[str], 
                          expected_chunks: List[List[int]], 
                          k: int = 5) -> Dict[str, Any]:
        """
        Evaluate retrieval performance using recall@k.
        
        Args:
            retriever: VectorStore instance with a search method.
            embedding_function: Function to get query embeddings.
            questions: List of test questions.
            expected_chunks: List of lists of expected chunk IDs for each question.
            k: Number of results to retrieve.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if len(questions) != len(expected_chunks):
            raise ValueError("Length of questions and expected_chunks must match")
        
        results = []
        
        for i, question in enumerate(questions):
            expected = set(expected_chunks[i])
            
            # Get query embedding
            query_embedding = embedding_function(question)
            
            # Get search results
            search_results = retriever.search(
                query_embedding=query_embedding,
                query_text=question,
                k=k
            )
            
            retrieved = set(result["metadata"].get("chunk_id") for result in search_results)
            
            # Calculate metrics
            correct = len(expected.intersection(retrieved))
            recall = correct / len(expected) if expected else 0
            precision = correct / len(retrieved) if retrieved else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                "question": question,
                "expected": list(expected),
                "retrieved": list(retrieved),
                "recall": recall,
                "precision": precision,
                "f1": f1
            })
        
        # Calculate average metrics
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        
        return {
            "results": results,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "avg_f1": avg_f1
        }