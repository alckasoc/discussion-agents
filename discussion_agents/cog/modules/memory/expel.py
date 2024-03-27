"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from scipy.spatial.distance import cosine

import tiktoken
from tiktoken.core import Encoding

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document

from discussion_agents.cog.modules.memory.base import BaseMemory
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class ExpeLExperienceMemory(BaseMemory):
    """ExpeL's experience pool memory.

    Attributes:
        experiences (Dict[str, List], optional): A dictionary storing experience data, 
            where each key is a task identifier. Generated from `gather_experience`.
        fewshot_questions (List[str], optional): A list of questions used in fewshot learning scenarios.
        fewshot_keys (List[str], optional): A list of answers (keys) corresponding to the fewshot questions.
        fewshot_examples (List[List[Tuple[str, str, str]]], optional): A nested list where each list 
            contains tuples of (thought, action, observation) used as fewshot examples.
        strategy (str): The strategy employed for handling and vectorizing experiences.
        embedder (Embeddings): An embedding object used for generating vector embeddings of documents.
        encoder (Encoding): An encoder object used for token counting within documents.
    """
    def __init__(
        self,
        experiences: Optional[Dict[str, List]] = {},
        fewshot_questions: Optional[List[str]] = [],
        fewshot_keys: Optional[List[str]] = [],
        fewshot_examples: Optional[List[List[Tuple[str, str, str]]]] = [],
        strategy: str = "task",
        embedder: Embeddings = HuggingFaceEmbeddings(),
        encoder: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initializes the memory with optional experiences, fewshot examples, and strategies.
        """
        super().__init__()

        self.experiences = deepcopy(experiences) if experiences else \
            {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
        self.fewshot_questions = fewshot_questions
        self.fewshot_keys = fewshot_keys
        self.fewshot_examples = fewshot_examples
        self.strategy = strategy
        self.embedder = embedder
        self.encoder = encoder

        # Collect all successful trajectories.
        success_traj_idxs = []
        if len(self.experiences['idxs']):
            success_traj_idxs = []
            for idx in self.experiences['idxs']:
                is_correct, _, _ = self.experiences['trajectories'][idx][0]  # Success on zero-th trial.
                if is_correct: success_traj_idxs.append(idx)

        self.success_traj_docs = []
        for idx in success_traj_idxs:
            question = self.experiences["questions"][idx]
            trajectory = self.experiences["trajectories"][idx][
                0
            ]  # Zero-th trial of trajectory.
            is_correct, _, steps = trajectory
            assert is_correct  # Ensure trajectory is successful.

            # Add the task.
            self.success_traj_docs.append(
                Document(
                    page_content=question, metadata={"type": "task", "task_idx": idx}
                )
            )

            # Add all trajectory actions.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=action,
                        metadata={"type": "action", "task_idx": idx},
                    )
                    for (_, action, _) in steps
                ]
            )

            # Add all trajectory thoughts.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=thought,
                        metadata={"type": "thought", "task_idx": idx},
                    )
                    for (thought, _, _) in steps
                ]
            )

            # Add each step.
            for step in steps:
                self.success_traj_docs.append(
                    Document(
                        page_content="\n".join(step),
                        metadata={"type": "step", "task_idx": idx},
                    )
                )

        # If including fewshot examples in experiences.
        if fewshot_questions and fewshot_keys and fewshot_examples:
            # Update self.experiences.
            for question, key, steps in zip(fewshot_questions, fewshot_keys, fewshot_examples):
                idx = max(self.experiences['idxs'], default=-1) + 1

                self.experiences['idxs'].append(idx)
                self.experiences['questions'].append(question)
                self.experiences['keys'].append(key)
                self.experiences['trajectories'].append(
                    [
                        (True, key, steps)
                    ]
                )
                self.experiences['reflections'].append([])

                # Update self.success_traj_docs.

                # Add the task.
                self.success_traj_docs.append(
                    Document(
                        page_content=question, metadata={
                            "type": "task", 
                            "task_idx": idx
                        }
                    )
                )

                # Add all trajectory actions.
                self.success_traj_docs.extend(
                    [
                        Document(
                            page_content=action,
                            metadata={"type": "action", "task_idx": idx},
                        )
                        for (_, action, _) in steps
                    ]
                )

                # Add all trajectory thoughts.
                self.success_traj_docs.extend(
                    [
                        Document(
                            page_content=thought,
                            metadata={"type": "thought", "task_idx": idx},
                        )
                        for (thought, _, _) in steps
                    ]
                )

                # Add each step.
                for step in steps:
                    self.success_traj_docs.append(
                        Document(
                            page_content="\n".join(step),
                            metadata={"type": "step", "task_idx": idx},
                        )
                    )

        # Create vectorstore.
        self.vectorstore = None
        if len(self.experiences['idxs']) and len(self.success_traj_docs):
            self.vectorstore = FAISS.from_documents(
                [doc for doc in self.success_traj_docs if doc.metadata['type'] == self.strategy], 
                self.embedder
            )

    def __len__(self) -> int:
        """Returns length of experiences."""
        return len(self.experiences['idxs'])

    def clear(self) -> None:
        """Clears all stored experiences from the memory.

        Resets the memory to its initial empty state.
        """
        self.experiences = {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
        self.success_traj_docs = []
        self.vectorstore = None

    def add_memories(
        self,
        questions: List[str],
        keys: List[str],
        trajectories: List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]],
        reflections: Optional[List[List[str]]] = [],
    ) -> None:
        """Adds new experiences to the memory, including associated questions, keys, 
        trajectories, and optional reflections.

        Args:
            questions (List[str]): Questions related to the experiences being added.
            keys (List[str]): Answers corresponding to the provided questions.
            trajectories (List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]]): A list of trajectories where each 
                trajectory is a list of tuples with a boolean indicating success, an action taken, and a list of steps.
            reflections (Optional[List[List[str]]], default=[]): A list of additional reflective notes on the experiences.
        """
        assert len(questions) == len(keys) == len(trajectories)

        if reflections:
            assert len(reflections) == len(questions)
        else:
            reflections = [[] for _ in range(len(questions))]

        start_idx = max(self.experiences["idxs"], default=-1) + 1

        # Update experiences.
        self.experiences["idxs"].extend(
            list(
                range(
                    start_idx,
                    start_idx + len(questions),
                )
            )
        )
        self.experiences["questions"].extend(questions)
        self.experiences["keys"].extend(keys)
        self.experiences["trajectories"].extend(trajectories)
        self.experiences["reflections"].extend(reflections)

        # Update success_traj_docs.
        success_traj_idxs = []
        for idx, trajectory in enumerate(trajectories, start_idx):
            is_correct, _, _ = trajectory[0]
            if is_correct: success_traj_idxs.append(idx)

        for idx in success_traj_idxs:
            question = self.experiences["questions"][idx]
            trajectory = self.experiences["trajectories"][idx][
                0
            ]  # Zero-th trial of trajectory.
            is_correct, _, steps = trajectory
            assert is_correct  # Ensure trajectory is successful.

            # Add the task.
            self.success_traj_docs.append(
                Document(
                    page_content=question, metadata={"type": "task", "task_idx": idx}
                )
            )

            # Add all trajectory actions.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=action,
                        metadata={"type": "action", "task_idx": idx},
                    )
                    for (_, action, _) in steps
                ]
            )

            # Add all trajectory thoughts.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=thought,
                        metadata={"type": "thought", "task_idx": idx},
                    )
                    for (thought, _, _) in steps
                ]
            )

            # Add each step.
            for step in steps:
                self.success_traj_docs.append(
                    Document(
                        page_content="\n".join(step),
                        metadata={"type": "step", "task_idx": idx},
                    )
                )

        if success_traj_idxs:
            # Create vectorstore.
            self.vectorstore = FAISS.from_documents(
                [doc for doc in self.success_traj_docs if doc.metadata['type'] == self.strategy], 
                self.embedder
            )

    def _fewshot_doc_token_count(self, fewshot_doc: Document) -> int:
        """Returns the token count of a given document's successful trajectory.
        
        Args:
            fewshot_doc (Document): The document containing trajectory data.
        
        Returns:
            int: The token count of the document's trajectory.
        """
        task_idx = fewshot_doc.metadata['task_idx']
        trajectory = self.experiences['trajectories'][task_idx]
        _, _, steps = trajectory[0]  # A successful trial.
        steps_str = "\n".join(["\n".join(step) for step in steps])
        return len(self.encoder.encode(steps_str))

    def load_memories(
        self, 
        query: str, 
        k_docs: int = 24, 
        num_fewshots: int = 6, 
        max_fewshot_tokens: int = 1500,
        reranker_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieves fewshot documents based on a similarity search, with optional re-ranking strategies.
        
        Args:
            query (str): The query to perform similarity search against.
            k_docs (int): The number of documents to return from a similarity search.
            num_fewshots (int): The number of fewshot examples to utilize or retrieve.
            max_fewshot_tokens (int): The maximum number of tokens allowed in a single fewshot example. Defaults to 1500.
            reranker_strategy (Optional[str]): The re-ranking strategy to be applied based on similarity measures.

        Returns:
            Dict[str, Any]: A dictionary of retrieved fewshot documents (strings).
        """

        # If empty.
        if not len(self.experiences['idxs']) or not k_docs or not num_fewshots or not max_fewshot_tokens:
            return {"fewshots": []}


        # Query the vectorstore.
        fewshot_docs = self.vectorstore.similarity_search(query, k=k_docs)

        # Post-processing.

        # Re-ranking, optional.
        if not reranker_strategy:
            fewshot_docs = list(fewshot_docs)
        elif reranker_strategy == 'length':
            fewshot_docs = list(sorted(fewshot_docs, key=self._fewshot_doc_token_count, reverse=True))
        elif reranker_strategy == 'thought':
            fewshot_tasks = set([doc.metadata['task_idx'] for doc in fewshot_docs])
            subset_docs = list(filter(lambda doc: doc.metadata['type'] == 'thought' and doc.metadata['task_idx'] in fewshot_tasks, list(self.success_traj_docs)))
            fewshot_docs = sorted(subset_docs, key=lambda doc: cosine(self.embedder.embed_query(doc.page_content), self.embedder.embed_query(query)))
        elif reranker_strategy == 'task':
            fewshot_tasks = set([doc.metadata['task_idx'] for doc in fewshot_docs])
            subset_docs = list(filter(lambda doc: doc.metadata['type'] == 'thought' and doc.metadata['task_idx'] in fewshot_tasks, list(self.success_traj_docs)))
            fewshot_docs = sorted(subset_docs, key=lambda doc: cosine(self.embedder.embed_query(doc.page_content), self.embedder.embed_query(query)))
        else:
            raise NotImplementedError
        
        current_tasks = set()
        fewshots = []

        # Filtering.
        # Exclude fewshot documents that exceed the token limit 
        # or have already been selected as fewshot examples to avoid redundancy.
        for fewshot_doc in fewshot_docs:
            task_idx = fewshot_doc.metadata["task_idx"]
            question = self.experiences['questions'][task_idx]
            trajectory = self.experiences['trajectories'][task_idx]
            _, _, steps = trajectory[0]  # Zero-th successful trial.
            steps = "\n".join(["\n".join(step) for step in steps])

            if len(self.encoder.encode(steps)) <= max_fewshot_tokens and \
                task_idx not in current_tasks:
                fewshots.append(f"{question}\n{steps}")
                current_tasks.add(task_idx)

            if len(fewshots) == num_fewshots:
                break

        return {"fewshots": fewshots}


    def show_memories(
        self, 
        experiences_key: str = "experiences",
        success_traj_docs_key: str = "success_traj_docs",
        vectorstore_key: str = "vectorstore"
    ) -> Dict[str, Any]:
        """Displays the current set of stored experiences and vectorstore information.
        
        Returns:
            Dict[str, Any]: A dictionary containing experiences, succcessful trajectory documents, and vectorstore details.
        """
        return {
            experiences_key: self.experiences,
            success_traj_docs_key: self.success_traj_docs,
            vectorstore_key: self.vectorstore
        }
    

class ExpeLInsightMemory(BaseMemory):
    """A memory management class for ExpeL insights, handling operations like adding, deleting,
    and updating insights within a memory storage with a maximum capacity.

    Attributes:
        insights (List[Dict[str, Any]]): A list to store insight dictionaries.
        max_num_insights (int): Maximum number of insights that can be stored.
    """
    def __init__(
        self, 
        insights: Optional[List[Dict[str, Any]]] = [],
        max_num_insights: int = 20,
        leeway: int = 5
    ) -> None:
        """Initializes the ExpeLInsightMemory with optional insights and a maximum storage limit.

        Args:
            insights (Optional[List[Dict[str, Any]]]): Initial list of insights to store in memory.
            max_num_insights (int): The maximum number of insights that can be stored.
            leeway (int): Number of memories allowed over max_num_insights before 
                delete_memories instantly deletes an indexed memory.
        """
        super().__init__()

        self.insights = deepcopy(insights)
        self.max_num_insights = max_num_insights
        self.leeway = leeway

    def __len__(self) -> int:
        """Returns length of insights."""
        return len(self.insights)

    def clear(self) -> None:
        """Clears all stored insights from the memory.
        """
        self.insights = []

    def add_memories(self, insights: List[Dict[str, Any]]) -> None:
        """Adds new insights to the memory, up to the maximum storage limit.

        Args:
            insights (List[Dict[str, Any]]): A list of insights to add to the memory.
        """
        self.insights.extend(insights)

    def delete_memories(self, idx: int) -> None:
        """Deletes an insight from memory based on its index. 
        
        Adjusts insight scores before deletion.

        Args:
            idx (int): The index of the insight to delete.
        """

        if idx < len(self.insights):
            if len(self.insights) >= self.max_num_insights + self.leeway:
                _ = self.insights.pop(idx)
            else:
                self.insights[idx]['score'] -= 1
                if self.insights[idx]['score'] <= 0:
                    _ = self.insights.pop(idx)

    def update_memories(self, idx: int, update_type: str, insight: Optional[str] = None) -> None:
        """Updates an insight or its score based on the specified update type.

        Args:
            idx (int): The index of the insight to update.
            update_type (str): The type of update ("EDIT", "AGREE").
            insight (str): The new insight text (if applicable). 
        """
        if update_type == "EDIT" and insight:
            self.insights[idx]['insight'] = insight
            self.insights[idx]['score'] += 1
        elif update_type == "AGREE":
            self.insights[idx]['score'] += 1
        else:
            raise NotImplementedError

    def load_memories(self, insights_key = "insights") -> Dict[str, Any]:
        """Loads and returns stored insights.

        Args:
            insights_key (str): The key name under which insights are returned.

        Returns:
            Dict[str, Any]: A dictionary containing stored insights.
        """
        return {insights_key: self.insights}

    def show_memories(self, insights_key = "insights") -> Dict[str, Any]:
        """Returns a dictionary of all stored insights for display or analysis.

        Args:
            insights_key (str): The key name under which insights are returned.

        Returns:
            Dict[str, Any]: A dictionary containing stored insights.
        """
        return {insights_key: self.insights}
