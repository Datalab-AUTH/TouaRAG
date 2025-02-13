"""
This module provides classes and functions for extracting triples from text using a language model, 
storing and analyzing property graphs in Neo4j, and querying these graphs with a focus on 
community detection and summarization.

Classes:
    GraphRAGExtractor:
        Extracts triples from a graph using a language model and a simple prompt + output parsing.
            llm (LLM): The language model to use.
            extract_prompt (Union[str, PromptTemplate]): The prompt to use for extracting triples.
            parse_fn (Callable): A function to parse the output of the language model.
            num_workers (int): The number of workers to use for parallel processing.
            max_paths_per_chunk (int): The maximum number of paths to extract per chunk.

    GraphRAGStore:
        Manages and analyzes a property graph stored in Neo4j, with a focus on 
        community detection and summarization.
            generate_community_summary(text): Generates a summary for a given text 
                                              using a language model.
            build_communities(): Builds communities from the graph and summarizes them.
            _create_nx_graph(): Converts the internal graph representation to a NetworkX graph.
            _collect_community_info(nx_graph, clusters): Collects information for each node based on 
                            their community, allowing entities to belong to multiple clusters.
            _summarize_communities(community_info): Generates and stores summaries for 
                                                    each community.
            get_community_summaries(): Returns the community summaries, building them if 
                                       not already done.

    GraphRAGQueryEngine:
        Custom query engine for processing community summaries to generate answers to 
        specific queries.
            graph_store (GraphRAGStore): The graph store to query.
            index (PropertyGraphIndex): The property graph index.
            llm (LLM): The language model to use.
            similarity_top_k (int): The number of top similar entities to retrieve.
            prompt (str): Custom prompt for generating final answers.
            custom_query(query_str): Processes all community summaries to generate answers to 
                                     a specific query.
            get_entities(query_str, similarity_top_k): Retrieves entities from the 
                                                       query string.
            retrieve_entity_communities(entity_info, entities): Retrieves cluster information 
                                                        for given entities.
            generate_answer_from_summary(community_summary, query): Generates an answer 
                                                from a community summary based on a given query.
            aggregate_answers(community_answers): Aggregates individual community answers 
                                                into a final, coherent response.
"""
import re
from collections import defaultdict
from typing import Any, List, Callable, Optional, Union
import asyncio

from touarag.llm.openai import load_base_openai_llm
import nest_asyncio


from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PropertyGraphIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, QueryType

import networkx as nx
from graspologic.partition import hierarchical_leiden



nest_asyncio.apply()

class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, 
    relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 8,
    ) -> None:
        """Init params."""

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    A class to manage and analyze a property graph stored in Neo4j, 
    with a focus on community detection and summarization.

    Attributes:
        community_summary (dict): A dictionary to store summaries of communities.
        entity_info (dict): Information about entities in the graph.
        max_cluster_size (int): The maximum size of clusters for community detection.
        reset (bool): A flag to indicate whether to reset the database on initialization.
    Methods:
        generate_community_summary(text):
            Generates a summary for a given text using a language model.
        build_communities():
            Builds communities from the graph and summarizes them.
        _create_nx_graph():
            Converts the internal graph representation to a NetworkX graph.
        _collect_community_info(nx_graph, clusters):
            Collects information for each node based on their community, allowing entities to 
            belong to multiple clusters.
        _summarize_communities(community_info):
            Generates and stores summaries for each community.
        get_community_summaries():
            Returns the community summaries, building them if not already done.
        reset_database():
            Resets the database by deleting all nodes and relationships.
    """
    community_summary = {}
    entity_info = defaultdict(set)
    max_cluster_size = 5
    community_summaries = {}

    def __init__(self, reset: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if reset:
            self.reset_database()

    def reset_database(self) -> None:
        """
        Reset the database by deleting all nodes and relationships.
        """
        self.structured_query("MATCH (n) DETACH DELETE n")

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        model = load_base_openai_llm()
        response = model.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

class GraphRAGQueryEngine(CustomQueryEngine):
    """    
    GraphRAGQueryEngine is a custom query engine designed to process community summaries
    and generate answers to specific queries using a combination of graph-based retrieval
    and language model generation.
    Attributes:
        graph_store (GraphRAGStore): The store containing graph-related data and 
                                    community summaries.
        index (PropertyGraphIndex): The index used for retrieving relevant nodes 
                                    based on similarity.
        llm (LLM): The language model used for generating answers from summaries.
        similarity_top_k (int): The number of top similar entities to retrieve.
        prompt (str): Optional custom prompt for aggregating answers.
    Methods:
        custom_query(query_str: str) -> str:
            Process all community summaries to generate answers to a specific query.
        get_entities(query_str: str, similarity_top_k: int) -> list:
            Retrieve entities from the graph based on the query string and similarity threshold.
        retrieve_entity_communities(entity_info: dict, entities: list) -> list:
        generate_answer_from_summary(community_summary: str, query: str) -> str:
            Generate an answer from a community summary based on a given query using LLM.
        aggregate_answers(community_answers: list) -> str:
            Aggregate individual community answers into a final, coherent response.
    """
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20
    prompt: str = None
    async def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        # Use the graph store to retrieve community summaries once (calculated at vector store init)
        community_summaries = self.graph_store.community_summaries

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )

        # Save retrieved community summaries separately
        selected_community_summaries = {
            id: community_summary
            for id, community_summary in community_summaries.items()
            if id in community_ids
        }

        # TODO: Make it run in parallel to save exec time (async) (5x time)
        community_answers = await asyncio.gather(
            *[
            self.generate_answer_from_summary(community_summary, query_str)
            for community_summary in selected_community_summaries.values()
            ]
        )

        final_answer = self.aggregate_answers(community_answers)
        return final_answer, list(selected_community_summaries.values())

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = (
            r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
        )

        for node in nodes_retrieved:
            matches = re.findall(
                pattern, node.text, re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    async def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        if self.prompt:
            prompt = self.prompt
        else:
            prompt = "Combine the following intermediate answers into a final, concise response."

        # Use LLM to generate the final response
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response
    
    def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        return asyncio.run(self._aquery(str_or_query_bundle))

    async def _aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            # if query bundle, just run the query
            if isinstance(str_or_query_bundle, QueryBundle):
                query_str = str_or_query_bundle.query_str
            else:
                query_str = str_or_query_bundle
            # Also take generated community summaries to use as retrieved context
            raw_response, source_nodes = await self.custom_query(query_str)
            return (
                Response(raw_response, source_nodes=source_nodes)
                if isinstance(raw_response, str)
                else raw_response
            )