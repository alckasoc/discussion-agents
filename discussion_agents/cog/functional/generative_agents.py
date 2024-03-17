"""Functional module for Generative Agents."""

import re

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.vectorstores import FAISS 

from langchain.chains import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.utils.format import format_memories_detail
from discussion_agents.utils.parse import parse_list


def _create_default_time_weighted_retriever(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    embedding_size: int = 768,
    model_kwargs: Dict[str, Any] = {"device": "cpu"},
    encode_kwargs: Dict[str, Any] = {"normalize_embeddings": False},
    k: int = 5,
) -> TimeWeightedVectorStoreRetriever:
    """Returns a TimeWeightedVectorStoreRetriever with customizable parameters.

    This function sets up a retriever for semantic search using sentence embeddings. It uses a HuggingFace embedding
    model, specified by `model_name`, to generate embeddings. The embeddings are stored in a FAISS index for efficient
    similarity searching in high-dimensional space, and the retrieval process is managed by an in-memory document store.

    Args:
        model_name (str): Name of the HuggingFace model to use for embeddings. Defaults to 'sentence-transformers/all-mpnet-base-v2'.
        embedding_size (int): The dimensionality of the embeddings. Defaults to 768.
        model_kwargs (Dict[str, Any]): Additional keyword arguments for the embedding model. Defaults to {"device": "cpu"}.
        encode_kwargs (Dict[str, Any]): Keyword arguments for the embedding encoding process. Defaults to {"normalize_embeddings": False}.
        k (int): The number of top documents to retrieve based on similarity and additional scoring. Defaults to 5.

    Returns:
        TimeWeightedVectorStoreRetriever: An instance of TimeWeightedVectorStoreRetriever with the specified configuration.
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=k
    )
    return retriever


def score_memories_importance(
    memory_contents: Union[str, List[str]],
    relevant_memories: Union[str, List[str]],
    llm: Any,
    importance_weight: float = 0.15,
) -> List[float]:
    """Calculate absolute importance scores for given memory contents.

    Args:
        memory_contents (Union[str, List[str]): Memories or memory contents to score.
        relevant_memories (Union[str, List[str]): Relevant memories to all memory_contents;
            if memory_contents and relevant_memories are both str/list, then they correspond 1-to-1;
            if memory_contents is str and relevant_memories is list, then the topic will use all relevant_memories;
            if memory_contents is list and relevant_memories is str, then relevant_memories is broadcasted to all memory_contents.
        llm (LLM): a LangChain LLM instance.
        importance_weight (float, optional): Weight for importance scores. Default is 0.15.

    Returns:
        List[float]: List of importance scores (1.0 to 10.0 scale normalized and weighted).
    """
    if isinstance(memory_contents, str) and isinstance(relevant_memories, str):
        memory_contents, relevant_memories = [memory_contents], [relevant_memories]
    elif isinstance(memory_contents, str) and isinstance(relevant_memories, list):
        memory_contents = [memory_contents]
        relevant_memories = ["\n".join(relevant_memories)]
    elif isinstance(memory_contents, list) and isinstance(relevant_memories, str):
        relevant_memories = [relevant_memories] * len(memory_contents)

    assert len(memory_contents) == len(relevant_memories)

    relevant_memories = "\n".join(relevant_memories)

    prompt = PromptTemplate.from_template(
        "On the scale of 1 to 10, where 1 is purely mundane (e.g., routine morning greetings) "
        + "and 10 is extremely poignant (e.g., a conversation about breaking up, a fight) "
        + ", rate the likely poignancy of the "
        + "following piece of memory with respect to these following relevant memories:\n"
        + "{relevant_memories}\n\n"
        + "Provide only a single rating.\n"
        + "\Memory: {memory_content}\n"
        + "Rating (return a number between 1 to 10): "
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    scores = []
    for i, (memory_content, relevant_memory) in enumerate(
        zip(memory_contents, relevant_memories)
    ):
        score = chain.run(
            relevant_memories=relevant_memory, memory_content=memory_content
        ).strip()
        score = re.findall(r"\d+", score)
        assert (
            len(score) == 1
        ), f"Found multiple scores in LLM output for memory_content at index {i}: {score}."
        scores.append(score[0])

    assert len(scores) == len(
        memory_contents
    ), f"The llm generated {len(scores)} scores for {len(memory_contents)} memories."

    # Normalize and convert to floats.
    scores = [float(x) / 10 * importance_weight for x in scores]

    return scores


def get_topics_of_reflection(
    observations: Union[str, List[str]],
    llm: Any,
) -> List[str]:
    """Generate three insightful high-level questions based on recent observation(s).

    Args:
        observations (Union[str, List[str]]): Recent observations to derive questions from; can be multiple but must be str.
        llm (LLM): a LangChain LLM instance.

    Returns:
        List[str]: A list of the three most salient high-level questions.

    Note:
        This function takes recent observations as input and utilizes the specified core
        component to generate meaningful and high-level questions related to the observed
        subjects.
    """
    if isinstance(observations, list):
        observations = "\n".join(observations)

    prompt = PromptTemplate.from_template(
        "{observations}\n\n"
        + "Given only the information above, what are the 3 most salient "
        + "high-level questions we can answer about the subjects in the statements?\n"
        + "Provide each question on a new line."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = parse_list(chain.run(observations=observations))
    return result


def get_insights_on_topics(
    topics: Union[str, List[str]],
    related_memories: Union[str, List[str]],
    llm: Any,
) -> List[List[str]]:
    """Generate high-level insights on specified topics using relevant memories.

    Args:
        topics (Union[str, List[str]]): A list of topics (or a str for 1 topic) for which insights are to be generated.
        related_memories (Union[str, List[str]]): Memories relevant to the specified topic(s);
            if topics and related_memories are both str/list, then they correspond 1-to-1;
            if topics is str and related_memories is list, then the topic will use all related_memories;
            if topics is list and related_memories is str, then related_memories is broadcasted to all topics.
        llm (LLM): a LangChain LLM instance.

    Returns:
        List[List[str]]: Lists of high-level, unique insights corresponding to each topic.

    Note:
        This function leverages pertinent memories associated with the specified topics
        and employs the provided core component to create novel, high-level insights
        based on the gathered information.
    """
    if isinstance(topics, str) and isinstance(related_memories, str):
        topics, related_memories = [topics], [related_memories]
    elif isinstance(topics, str) and isinstance(related_memories, list):
        topics = [topics]
        related_memories = ["\n".join(related_memories)]
    elif isinstance(topics, list) and isinstance(related_memories, str):
        related_memories = [related_memories] * len(topics)

    assert type(topics) == type(related_memories) == list
    assert len(topics) == len(related_memories)

    prompt = PromptTemplate.from_template(
        "Statements relevant to: '{topic}'\n"
        + "---\n"
        + "{related_memory}\n"
        + "---\n"
        + "What 5 high-level novel insights can you infer from the above statements "
        + "that are relevant to answering the following question?\n"
        + "Do not include any insights that are not relevant to the question.\n"
        + "Do not repeat any insights that have already been made.\n\n"
        + "Question: {topic}\n\n"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    results = []
    for topic, related_memory in zip(topics, related_memories):
        result = chain.run(topic=topic, related_memory=related_memory)
        results.append(parse_list(result))

    return results


def reflect(
    observations: Union[str, List[str]],
    llm: Any,
    retriever: BaseRetriever,
    now: Optional[datetime] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Generate insights on recent observations through reflection.

    This function generates a list of topics w.r.t observations and extracts
    salient insights from each of these topics using related memories (to these topics)
    as context. Related memories are specific to each topic/observation.

    Args:
        observations (Union[str, List[str]]): Observations to derive reflections from.
        llm (LLM): a LangChain LLM instance.
        retriever (BaseRetriever): A BaseRetriever to extract relevant memories.
        now (Optional[datetime]): current datetime or one specified.

    Returns:
        Tuple[List[str], List[List[str]]]: A list of generated topics from the
            observation and a list of lists of insights, a list of insights for
            every topic.
    """
    topics = get_topics_of_reflection(observations=observations, llm=llm)

    related_memories = []
    for topic in topics:
        fetched_memories = fetch_memories(
            observation=topic, memory_retriever=retriever, now=now
        )
        topic_related_memories = "\n".join(
            [
                format_memories_detail(memories=memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(fetched_memories)
            ]
        )
        related_memories.append(topic_related_memories)

    insights = get_insights_on_topics(
        topics=topics, related_memories=related_memories, llm=llm
    )
    return topics, insights
