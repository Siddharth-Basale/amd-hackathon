"""
VectorizerE - Graph-Based Markdown Vectorizer
Takes .md file input, builds a knowledge graph, and vectorizes chunks.
Uses Ollama embeddings (nomic-embed-text:v1.5) for vectorization.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re
import urllib.request
import urllib.error
# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available. Install with: pip install networkx")

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from knowledge.entity_extractor import EntityExtractor, ExtractionResult
from knowledge import graph_builder

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# ---------------- CONFIGURATION ---------------- 
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
OLLAMA_BASE_URL = "http://localhost:11434"
ENTITY_EXTRACTION_MODEL = os.getenv("PLAN_E_ENTITY_MODEL", "gpt-4o-mini")
ENTITY_EXTRACTION_INTENSITY_THRESHOLD = int(os.getenv("PLAN_E_EXTRACTION_INTENSITY_THRESHOLD_CHARS", "3000"))
ENABLE_ENTITY_EXTRACTION = os.getenv("PLAN_E_ENABLE_ENTITY_EXTRACTION", "1").lower() not in {"0", "false", "no"}

# Token limits
EMBEDDING_MAX_TOKENS = 1500

# Character limits
EMBEDDING_MAX_CHARS = 2000
CHUNK_SIZE = EMBEDDING_MAX_CHARS
CHUNK_OVERLAP = 150

# Adaptive chunking for small documents (env overrides)
MIN_CHUNKS = int(os.getenv("PLAN_E_MIN_CHUNKS", "8"))  # Target minimum chunks for short docs
MIN_CHUNK_CHARS = int(os.getenv("PLAN_E_MIN_CHUNK_CHARS", "300"))  # Floor to avoid tiny chunks

# Token counting encoding
TOKEN_ENCODING = "cl100k_base" if TIKTOKEN_AVAILABLE else None

# ---------------- TOKEN TRACKING ---------------- 
class TokenTracker:
    """Track token usage for different models"""
    
    def __init__(self):
        self.encoding = None
        if TIKTOKEN_AVAILABLE and TOKEN_ENCODING:
            try:
                self.encoding = tiktoken.get_encoding(TOKEN_ENCODING)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
        
        self.stats = {
            "embedding_tokens": 0,
            "total_chunks": 0,
            "truncated_chunks": 0
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken (approximate for WordPiece models)"""
        if not self.encoding:
            return int(len(text) / 1.5)
        
        try:
            tiktoken_count = len(self.encoding.encode(text))
            wordpiece_estimate = int(tiktoken_count * 1.3)
            return wordpiece_estimate
        except Exception as e:
            logger.warning(f"Token counting error: {e}, using fallback")
            return int(len(text) / 1.5)
    
    def check_embedding_limit(self, text: str) -> Tuple[str, bool]:
        """Check if text exceeds embedding limit, truncate if needed"""
        if len(text) > EMBEDDING_MAX_CHARS:
            self.stats["truncated_chunks"] += 1
            truncated = text[:EMBEDDING_MAX_CHARS]
            last_period = truncated.rfind('. ')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > len(truncated) * 0.7:
                truncated = truncated[:cut_point + 1] + "..."
            else:
                truncated = truncated + "..."
            
            tokens = self.count_tokens(truncated)
            self.stats["embedding_tokens"] += tokens
            if tokens > EMBEDDING_MAX_TOKENS:
                logger.warning(f"After char truncation, still {tokens} tokens")
                truncated = self._truncate_to_tokens(truncated, int(EMBEDDING_MAX_TOKENS * 0.9))
            return truncated, True
        
        tokens = self.count_tokens(text)
        SAFETY_MARGIN = int(EMBEDDING_MAX_TOKENS * 0.85)
        
        if tokens > SAFETY_MARGIN:
            self.stats["truncated_chunks"] += 1
            truncated = self._truncate_to_tokens(text, SAFETY_MARGIN)
            new_tokens = self.count_tokens(truncated)
            self.stats["embedding_tokens"] += new_tokens
            if new_tokens > EMBEDDING_MAX_TOKENS:
                logger.warning(f"Truncation may not be sufficient: {new_tokens} tokens")
            return truncated, True
        
        self.stats["embedding_tokens"] += tokens
        return text, False
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not self.encoding:
            max_chars = int(max_tokens * 1.5)
            if len(text) <= max_chars:
                return text
            truncated = text[:max_chars]
            last_period = truncated.rfind('. ')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > max_chars * 0.7:
                return truncated[:cut_point + 1] + "..."
            return truncated + "..."
        
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            safe_max = int(max_tokens * 0.95)
            truncated_tokens = tokens[:safe_max]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            verify_tokens = self.encoding.encode(truncated_text)
            if len(verify_tokens) > max_tokens:
                truncated_tokens = tokens[:int(max_tokens * 0.9)]
                truncated_text = self.encoding.decode(truncated_tokens)
            
            last_period = truncated_text.rfind('. ')
            last_newline = truncated_text.rfind('\n')
            cut_point = max(last_period, last_newline)
            
            if cut_point > len(truncated_text) * 0.7:
                final_text = truncated_text[:cut_point + 1] + "..."
            else:
                final_text = truncated_text + "..."
            
            final_tokens = self.encoding.encode(final_text)
            if len(final_tokens) > max_tokens:
                hard_truncated = tokens[:int(max_tokens * 0.9)]
                final_text = self.encoding.decode(hard_truncated) + "..."
            
            return final_text
        except Exception as e:
            logger.warning(f"Token truncation error: {e}, using character truncation")
            max_chars = int(max_tokens * 1.5)
            return text[:max_chars] + "..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset stats for a new vectorization run."""
        self.stats = {
            "embedding_tokens": 0,
            "total_chunks": 0,
            "truncated_chunks": 0,
        }

# Global token tracker
token_tracker = TokenTracker()

# ---------------- DOCUMENT GRAPH ---------------- 
class DocumentGraph:
    """Knowledge graph for document structure"""
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for Plan E. Install with: pip install networkx")
        
        self.graph = nx.DiGraph()  # Directed graph
        self.chunk_nodes = {}  # chunk_id -> node_id
        self.section_nodes = {}  # section_path -> node_id
    
    def add_section_node(self, section_path: str, section_title: str, level: int, start_line: int):
        """Add a section node to the graph"""
        node_id = f"section:{section_path}"
        if node_id not in self.graph:
            self.graph.add_node(node_id, 
                              type="section",
                              section_path=section_path,
                              section_title=section_title,
                              level=level,
                              start_line=start_line)
            self.section_nodes[section_path] = node_id
        return node_id
    
    def add_chunk_node(self, chunk_id: int, chunk: Document):
        """Add a chunk node to the graph"""
        node_id = f"chunk:{chunk_id}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="chunk",
                              chunk_id=chunk_id,
                              heading=chunk.metadata.get("heading", ""),
                              section_path=chunk.metadata.get("section_path", ""))
            self.chunk_nodes[chunk_id] = node_id
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, relation: str, **kwargs):
        """Add an edge between nodes"""
        if source_id and target_id:
            self.graph.add_edge(source_id, target_id, relation=relation, **kwargs)
    
    def get_parent_section(self, chunk_id: int) -> Optional[str]:
        """Get the parent section node for a chunk"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return None
        
        # Find section node connected via "belongs_to"
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "section":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "belongs_to":
                    return predecessor
        
        return None
    
    def get_section_chunks(self, section_path: str) -> List[int]:
        """Get all chunk IDs in a section"""
        section_node = self.section_nodes.get(section_path)
        if not section_node:
            return []
        
        chunk_ids = []
        for successor in self.graph.successors(section_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(section_node, successor)
                if edge_data and edge_data.get("relation") == "contains":
                    chunk_id = self.graph.nodes[successor].get("chunk_id")
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
        
        return sorted(chunk_ids)
    
    def get_adjacent_chunks(self, chunk_id: int, window: int = 1) -> List[int]:
        """Get adjacent chunks via 'follows' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        adjacent_ids = []
        
        # Get previous chunks (incoming "follows" edges)
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "follows":
                    prev_id = self.graph.nodes[predecessor].get("chunk_id")
                    if prev_id is not None:
                        adjacent_ids.append(prev_id)
        
        # Get next chunks (outgoing "follows" edges)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "follows":
                    next_id = self.graph.nodes[successor].get("chunk_id")
                    if next_id is not None:
                        adjacent_ids.append(next_id)
        
        return adjacent_ids[:window * 2]  # window chunks before + window chunks after
    
    def get_similar_chunks(self, chunk_id: int) -> List[int]:
        """Get chunks connected via 'similar_to' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        similar_ids = []
        
        # Get chunks connected via similarity edges (both directions)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[successor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[predecessor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        return similar_ids
    
    def expand_from_chunks(self, chunk_ids: List[int], max_expansion: int = 20) -> List[int]:
        """Expand retrieval by following graph edges from seed chunks"""
        expanded_ids = set(chunk_ids)
        to_process = list(chunk_ids)
        
        while to_process and len(expanded_ids) < max_expansion:
            current_id = to_process.pop(0)
            
            # Get parent section and all chunks in that section
            section_node = self.get_parent_section(current_id)
            if section_node:
                section_path = self.graph.nodes[section_node].get("section_path", "")
                section_chunks = self.get_section_chunks(section_path)
                for chunk_id in section_chunks:
                    if chunk_id not in expanded_ids:
                        expanded_ids.add(chunk_id)
                        to_process.append(chunk_id)
            
            # Get adjacent chunks
            adjacent = self.get_adjacent_chunks(current_id, window=1)
            for adj_id in adjacent:
                if adj_id not in expanded_ids:
                    expanded_ids.add(adj_id)
                    to_process.append(adj_id)
            
            # Get similar chunks via similarity edges
            similar = self.get_similar_chunks(current_id)
            for sim_id in similar:
                if sim_id not in expanded_ids:
                    expanded_ids.add(sim_id)
                    to_process.append(sim_id)
        
        return list(expanded_ids)
    
    def save(self, filepath: Path):
        """Save graph to JSON format"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                **node_data
            })
        
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    def load(self, filepath: Path):
        """Load graph from JSON format"""
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self.chunk_nodes.clear()
        self.section_nodes.clear()
        
        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
            
            # Rebuild index
            if node_data.get("type") == "chunk":
                chunk_id = node_data.get("chunk_id")
                if chunk_id is not None:
                    self.chunk_nodes[chunk_id] = node_id
            elif node_data.get("type") == "section":
                section_path = node_data.get("section_path", "")
                if section_path:
                    self.section_nodes[section_path] = node_id
        
        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

# ---------------- DOCUMENT STRUCTURE EXTRACTION ---------------- 
def extract_document_structure(markdown_content: str) -> Dict[str, Any]:
    """Extract document structure: sections, headers, tables"""
    lines = markdown_content.split("\n")
    
    structure = {
        "sections": [],
        "headers": [],
        "tables": []
    }
    
    current_section_path = []  # List of (level, title) tuples to maintain hierarchy
    line_number = 0
    in_table = False
    table_start = None
    
    for line in lines:
        line_number += 1
        
        # Detect tables
        if line.strip().startswith("|") and "|" in line:
            if not in_table:
                in_table = True
                table_start = line_number
        else:
            if in_table:
                structure["tables"].append({
                    "start_line": table_start,
                    "end_line": line_number - 1,
                    "section": current_section_path[-1][1] if current_section_path else None
                })
                in_table = False
        
        # Detect headers
        header_match = re.match(r'^(#+)\s+(.+)$', line.strip())
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            # Build hierarchical section path - keep full hierarchy
            # Remove any headings at the same or deeper level
            # Keep only headings at shallower levels (lower level numbers)
            current_section_path = [(l, t) for l, t in current_section_path if l < level]
            
            # Build full hierarchical path from all parent sections
            # Example: if we have # Part A, ## Part B, ### Section 1
            # For ### Section 1, path will be "Part A > Part B > Section 1" (full hierarchy)
            if current_section_path:
                # Build path from all parent sections
                parent_titles = [t for l, t in current_section_path]
                section_path_str = " > ".join(parent_titles + [title])
            else:
                # No parent sections, just use the title
                section_path_str = title
            
            # Update the current section path for tracking (keep full hierarchy)
            current_section_path.append((level, title))
            
            header_info = {
                "level": level,
                "title": title,
                "line": line_number,
                "section_path": section_path_str
            }
            structure["headers"].append(header_info)
            
            current_section = {
                "path": section_path_str,
                "title": title,
                "level": level,
                "start_line": line_number
            }
            structure["sections"].append(current_section)
    
    return structure

def get_section_for_line(line_number: int, structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the most specific (deepest) section that contains a given line number.
    
    This finds the section with the highest level (most nested) that starts before or at the line number.
    This ensures chunks are assigned to the most specific section they belong to.
    """
    sections = structure["sections"]
    
    # Find all sections that contain this line (start_line <= line_number)
    containing_sections = [s for s in sections if s["start_line"] <= line_number]
    
    if not containing_sections:
        return None
    
    # Find the section with the highest level (most nested/deepest)
    # If levels are equal, prefer the one that starts closest to the line (most recent)
    most_specific = max(containing_sections, key=lambda s: (s.get("level", 0), s["start_line"]))
    
    return most_specific

# ---------------- ENHANCED CHUNKING ---------------- 
def is_chunk_empty(chunk: Document) -> bool:
    """Check if a chunk is empty or has minimal content (only headers, whitespace, etc.)"""
    content = chunk.page_content.strip()
    
    # Empty or only whitespace
    if not content or len(content) < 10:
        return True
    
    # Count non-header, non-whitespace lines
    lines = content.split("\n")
    meaningful_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, headers, and horizontal rules
        if stripped and not stripped.startswith("#") and not re.match(r'^[-=*]{3,}$', stripped):
            meaningful_lines.append(stripped)
    
    # If we have less than 2 meaningful lines or very short content, consider it empty
    if len(meaningful_lines) < 2:
        return True
    
    # Check if meaningful content is too short (less than 50 chars of actual text)
    meaningful_text = " ".join(meaningful_lines)
    if len(meaningful_text) < 50:
        return True
    
    return False


def compute_effective_chunk_size(total_chars: int) -> int:
    """
    Adaptive chunk size by document length.
    Short docs get smaller chunks (more granular retrieval); long docs use fixed max.
    """
    effective = min(
        CHUNK_SIZE,
        max(MIN_CHUNK_CHARS, total_chars // MIN_CHUNKS),
    )
    return effective


def parse_markdown_enhanced(markdown_content: str) -> Tuple[List[Document], Dict[str, Any]]:
    """Parse markdown with structure awareness and enhanced metadata"""
    logger.info("Extracting document structure...")
    structure = extract_document_structure(markdown_content)
    
    level_counts = {}
    for header in structure['headers']:
        level = header.get('level', 0)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    logger.info(f"Found {len(structure['sections'])} sections, {len(structure['headers'])} headers, {len(structure['tables'])} tables")
    if level_counts:
        level_summary = ", ".join([f"Level {k}: {v}" for k, v in sorted(level_counts.items())])
        logger.info(f"Heading level distribution: {level_summary}")

    total_chars = len(markdown_content)
    effective_chunk_size = compute_effective_chunk_size(total_chars)
    if effective_chunk_size != CHUNK_SIZE:
        logger.info(
            "Adaptive chunking: doc has %d chars, using effective_chunk_size=%d (target min %d chunks)",
            total_chars, effective_chunk_size, MIN_CHUNKS,
        )
    
    lines = markdown_content.split("\n")
    chunks = []
    current_chunk_lines = []
    current_chunk_size = 0
    chunk_index = 0
    current_section = None
    current_section_path = ""
    
    for line_num, line in enumerate(lines, 1):
        # Update current section as we process lines
        section = get_section_for_line(line_num, structure)
        if section and section != current_section:
            current_section = section
            current_section_path = section["path"]
        
        is_header = bool(re.match(r'^#+\s+', line.strip()))
        
        line_with_newline = line + "\n"
        line_size = len(line_with_newline)
        
        if current_chunk_size + line_size > effective_chunk_size and current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            chunk_start_line = line_num - len(current_chunk_lines)
            
            # Get the section for the CHUNK'S START LINE (not current line)
            # This ensures chunks are assigned to the correct section
            chunk_section = get_section_for_line(chunk_start_line, structure)
            chunk_section_path = chunk_section["path"] if chunk_section else ""
            
            chunk = create_enhanced_chunk(
                chunk_text,
                chunk_index,
                chunk_section_path,
                chunk_section,
                structure,
                chunk_start_line
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Calculate minimal overlap: use only 1-2 lines for context
            # Overlap should be minimal to avoid duplicate content
            # Use maximum 2 lines or ~5% of chunk, whichever is smaller
            overlap_size = min(max(1, len(current_chunk_lines) // 20), 2)
            overlap_lines = current_chunk_lines[-overlap_size:] if len(current_chunk_lines) > overlap_size else []
            
            # Reset for next chunk with minimal overlap
            current_chunk_lines = overlap_lines
            current_chunk_size = sum(len(l) for l in current_chunk_lines)
        
        current_chunk_lines.append(line_with_newline)
        current_chunk_size += line_size
    
    if current_chunk_lines:
        chunk_text = "".join(current_chunk_lines)
        chunk_start_line = len(lines) - len(current_chunk_lines)
        
        # Get the section for the CHUNK'S START LINE
        chunk_section = get_section_for_line(chunk_start_line, structure)
        chunk_section_path = chunk_section["path"] if chunk_section else ""
        
        chunk = create_enhanced_chunk(
            chunk_text,
            chunk_index,
            chunk_section_path,
            chunk_section,
            structure,
            chunk_start_line
        )
        chunks.append(chunk)
    
    # Filter out empty chunks and duplicate chunks before adding adjacency metadata
    filtered_chunks = []
    seen_content_hashes = set()  # Track content to avoid duplicates
    
    for chunk in chunks:
        if is_chunk_empty(chunk):
            logger.debug(f"Skipping empty chunk: {chunk.metadata.get('heading', 'Unknown')}")
            continue
        
        # Check for duplicate/similar content (90%+ overlap)
        # Normalize: remove extra whitespace for comparison
        content_normalized = " ".join(chunk.page_content.split())
        
        # Check against all previously seen chunks for high similarity
        is_duplicate = False
        for seen_content in seen_content_hashes:
            # Compare first 1500 chars (catches most of chunk content)
            current_preview = content_normalized[:1500]
            seen_preview = seen_content[:1500]
            
            # Calculate word-based similarity (Jaccard similarity)
            current_words = set(current_preview.split())
            seen_words = set(seen_preview.split())
            
            if len(current_words) > 0 and len(seen_words) > 0:
                # Jaccard similarity: intersection over union
                intersection = len(current_words & seen_words)
                union = len(current_words | seen_words)
                similarity = intersection / union if union > 0 else 0.0
                
                # If 90%+ similar, consider it a duplicate
                if similarity >= 0.90:
                    logger.debug(f"Skipping duplicate chunk (similarity: {similarity:.1%}): {chunk.metadata.get('heading', 'Unknown')} (start_line: {chunk.metadata.get('start_line')})")
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
        
        # Store normalized content for future comparisons
        seen_content_hashes.add(content_normalized)
        filtered_chunks.append(chunk)
    
    # Update chunk indices after filtering
    for i, chunk in enumerate(filtered_chunks):
        chunk.metadata["chunk_index"] = i
    
    # Add adjacency metadata
    for i, chunk in enumerate(filtered_chunks):
        chunk.metadata["prev_chunk_ids"] = [i-2, i-1] if i >= 2 else [i-1] if i >= 1 else []
        chunk.metadata["next_chunk_ids"] = [i+1, i+2] if i < len(filtered_chunks)-2 else [i+1] if i < len(filtered_chunks)-1 else []
        chunk.metadata["sequential_position"] = i
    
    logger.info(f"Created {len(filtered_chunks)} chunks (filtered {len(chunks) - len(filtered_chunks)} empty chunks)")
    return filtered_chunks, structure

def create_enhanced_chunk(
    text: str,
    chunk_index: int,
    section_path: str,
    section: Optional[Dict[str, Any]],
    structure: Dict[str, Any],
    start_line: int
) -> Document:
    """Create a chunk with enhanced metadata"""
    
    heading = ""
    lines = text.split("\n")
    
    # Strategy 1: If we have section information, prefer the section's title
    # This ensures chunks get the correct heading from their assigned section
    if section:
        section_title = section.get("title", "")
        if section_title:
            heading = section_title
    
    # Strategy 2: If no section title, try to find a header in the chunk content
    # Look through the entire chunk for any markdown header (all levels: #, ##, ###, ####, etc.)
    if not heading:
        for line in lines:
            line_stripped = line.strip()
            # Match any markdown header: #, ##, ###, ####, etc.
            header_match = re.match(r'^(#+)\s+(.+)$', line_stripped)
            if header_match:
                heading = header_match.group(2).strip()
                break  # Use the first header found
    
    # Strategy 3: If still no heading, use the last part of section_path (most specific heading)
    if not heading and section_path:
        path_parts = section_path.split(" > ")
        heading = path_parts[-1] if path_parts else ""
    
    # Strategy 4: Only as last resort, use generic chunk name
    if not heading:
        heading = f"Chunk {chunk_index + 1}"
    
    has_table = any("|" in line and line.strip().startswith("|") for line in lines)
    
    table_context = None
    if has_table:
        table_lines = [line for line in lines if "|" in line and line.strip().startswith("|")]
        if table_lines:
            table_context = "\n".join(table_lines[:3])
    
    metadata = {
        "chunk_index": chunk_index,
        "chunk_size": len(text),
        "heading": heading,
        "section_path": section_path or "",
        "section_title": section["title"] if section else "",
        "section_level": section["level"] if section else 0,
        "chunk_type": "table" if has_table else "text",
        "table_context": table_context,
        "start_line": start_line,
        "has_table": has_table,
    }
    
    return Document(page_content=text, metadata=metadata)

# ---------------- STATE DEFINITION ---------------- 
class VectorizerState(TypedDict):
    """State for the vectorization workflow"""
    markdown_file: str
    document_path: str
    chunks: List[Document]
    structure: Dict[str, Any]
    processed_chunks: List[Dict[str, Any]]
    vector_store: Any
    document_graph: DocumentGraph
    json_mapping: List[Dict[str, Any]]
    output_folder: Optional[str]
    output_root: Optional[str]
    document_output_dir: Optional[str]
    token_usage: Optional[Dict[str, Any]]

# ---------------- WORKFLOW NODES ---------------- 
def load_markdown(state: VectorizerState) -> VectorizerState:
    """Load and parse markdown file"""
    md_path = Path(state["markdown_file"])
    
    if md_path.is_dir():
        md_files = list(md_path.glob("*.md"))
        if not md_files:
            logger.error(f"No .md file found in folder: {md_path}")
            raise FileNotFoundError(f"No .md file found in {md_path}")
        md_path = md_files[0]
        logger.info(f"Found markdown file: {md_path.name}")
    elif md_path.suffix.lower() != ".md":
        raise ValueError(f"Input must be a .md file or folder containing .md, got: {md_path}")
    
    logger.info(f"Loading markdown file: {md_path}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    chunks, structure = parse_markdown_enhanced(markdown_content)
    
    state["chunks"] = chunks
    state["structure"] = structure
    state["document_path"] = str(md_path)
    state["output_folder"] = str(md_path.parent)
    if not state.get("output_root"):
        state["output_root"] = str(Path("output"))
    logger.info(f"Loaded and chunked {len(chunks)} chunks with structure awareness")
    return state

def process_chunks_one_by_one(state: VectorizerState) -> VectorizerState:
    """Process chunks: summarize, embed, and build graph"""
    chunks = state["chunks"]
    structure = state["structure"]
    # Reset token stats so this run's usage is logged correctly (economics tracker)
    token_tracker.reset_stats()
    
    document_path_value = state.get("document_path") or state.get("markdown_file")
    if not document_path_value:
        raise ValueError("Document path not found in state")
    document_path = Path(document_path_value)
    doc_stem = document_path.stem

    output_root_value = state.get("output_root") or "output"
    output_root = Path(output_root_value)
    output_root.mkdir(parents=True, exist_ok=True)

    document_output_dir = output_root / doc_stem
    document_output_dir.mkdir(parents=True, exist_ok=True)

    vector_db_dir = document_output_dir / "vector_db" / doc_stem
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = document_output_dir / f"{doc_stem}_vector_mapping.json"
    graph_output_path = document_output_dir / f"{doc_stem}_document_graph.json"
    
    # Initialize embeddings
    logger.info("Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Initialize vector store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_dir)
    )
    
    # Initialize document graph
    document_graph = DocumentGraph()
    state["document_output_dir"] = str(document_output_dir)
    
    # Build graph structure - add ALL sections as nodes (not just those with chunks)
    # This ensures every heading becomes a node in the graph, preserving full hierarchy
    logger.info("Building document graph...")
    
    # Track which sections have chunks (for logging)
    sections_with_chunks = set()
    for chunk in chunks:
        section_path = chunk.metadata.get("section_path", "")
        if section_path:
            sections_with_chunks.add(section_path)
    
    # Build a map of all section paths to section info
    all_sections_map = {}
    for section in structure["sections"]:
        all_sections_map[section["path"]] = section
    
    # Add ALL sections from the structure as nodes (every heading should be a node)
    # This ensures complete hierarchical representation
    sections_to_add = set()
    for section in structure["sections"]:
        section_path = section["path"]
        sections_to_add.add(section_path)
        
        # Also add all parent sections in the hierarchy path
        if " > " in section_path:
            path_parts = section_path.split(" > ")
            current_path = ""
            for i, part in enumerate(path_parts):
                if i == 0:
                    current_path = part
                else:
                    current_path = current_path + " > " + part
                sections_to_add.add(current_path)
    
    # Add all section nodes (every heading becomes a node)
    for section_path in sorted(sections_to_add):
        if section_path in all_sections_map:
            section = all_sections_map[section_path]
            section_node = document_graph.add_section_node(
                section["path"],
                section["title"],
                section["level"],
                section["start_line"]
            )
        else:
            # Parent section not in structure (shouldn't happen, but handle gracefully)
            # This can happen if a parent section exists in the hierarchy but wasn't parsed
            # Extract the title from the path
            path_parts = section_path.split(" > ")
            title = path_parts[-1]
            
            # Determine level based on position in hierarchy (first part = level 2, second = level 3, etc.)
            # Level 1 = #, Level 2 = ##, Level 3 = ###, etc.
            level = len(path_parts) + 1  # First part is level 2 (##)
            
            # Try to find level from a similar section in structure
            for s in structure["sections"]:
                if s["path"] == section_path or s["title"] == title:
                    level = s["level"]
                    break
            
            # Try to find start_line from structure
            start_line = 0
            for s in structure["sections"]:
                if s["title"] == title:
                    start_line = s["start_line"]
                    break
            
            section_node = document_graph.add_section_node(
                section_path,
                title,
                level,
                start_line
            )
    
    # Create parent-child edges between sections
    logger.info("Creating parent-child relationships between sections...")
    section_edges_added = 0
    for section_path in sections_to_add:
        # Check if this section has a parent
        if " > " in section_path:
            # Extract parent path (everything except the last part)
            path_parts = section_path.split(" > ")
            parent_path = " > ".join(path_parts[:-1])
            
            # If parent section exists, create edge
            if parent_path in document_graph.section_nodes:
                child_node = document_graph.section_nodes.get(section_path)
                parent_node = document_graph.section_nodes.get(parent_path)
                if child_node and parent_node:
                    document_graph.add_edge(parent_node, child_node, relation="contains_section")
                    section_edges_added += 1
    
    logger.info(f"Created {len(sections_to_add)} section nodes (all headings from document structure)")
    logger.info(f"  - Sections with chunks: {len(sections_with_chunks)}")
    logger.info(f"  - Total sections in document: {len(structure['sections'])}")
    logger.info(f"Created {section_edges_added} parent-child section edges")
    
    # Process chunks
    json_mapping = []
    processed_chunks = []
    
    logger.info(f"Processing {len(chunks)} chunks with graph structure...")

    entity_extractor: Optional[EntityExtractor] = None
    knowledge_records: List[Dict[str, Any]] = []
    if ENABLE_ENTITY_EXTRACTION:
        try:
            total_doc_chars = sum(len(c.page_content) for c in chunks)
            extraction_intensity = (
                "minimal" if total_doc_chars > ENTITY_EXTRACTION_INTENSITY_THRESHOLD else "moderate"
            )
            entity_extractor = EntityExtractor(
                model=ENTITY_EXTRACTION_MODEL,
                temperature=0.0,
                extraction_intensity=extraction_intensity,
            )
            logger.info(
                "Entity extraction enabled using model %s, intensity %s (doc %d chars, threshold %d)",
                ENTITY_EXTRACTION_MODEL,
                extraction_intensity,
                total_doc_chars,
                ENTITY_EXTRACTION_INTENSITY_THRESHOLD,
            )
        except Exception as exc:
            logger.warning(
                "Entity extraction disabled: failed to initialise extractor (%s)", exc
            )
            entity_extractor = None
    else:
        logger.info("Entity extraction disabled by configuration")
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks (double-check, though they should already be filtered)
        if is_chunk_empty(chunk):
            logger.debug(f"Skipping empty chunk {i} during processing")
            continue
        
        # Use i as the vector_number (actual position after filtering)
        vector_number = i
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
        
        # Get heading from metadata, prefer section information over generic chunk names
        heading = chunk.metadata.get("heading", "")
        section_title = chunk.metadata.get("section_title", "")
        section_path = chunk.metadata.get("section_path", "")
        
        # If heading is empty or is a fallback pattern like "Chunk X", try to use section information
        if not heading or heading.startswith("Chunk "):
            # Prefer section_title, then last part of section_path, then generic name
            if section_title:
                heading = section_title
            elif section_path:
                path_parts = section_path.split(" > ")
                heading = path_parts[-1] if path_parts else f"Chunk {vector_number}"
            else:
                heading = f"Chunk {vector_number}"
        # If heading is a real heading (not "Chunk X" pattern), keep it as is
        
        section_path = chunk.metadata.get("section_path", "")
        
        # Prepare content for embedding
        content = chunk.page_content
        original_length = len(content)
        content, was_truncated = token_tracker.check_embedding_limit(content)
        
        # Prepare metadata for Chroma
        prev_chunk_ids = chunk.metadata.get("prev_chunk_ids", [])
        next_chunk_ids = chunk.metadata.get("next_chunk_ids", [])
        
        chroma_metadata = {
            "vector_number": vector_number,
            "chunk_index": vector_number,
            "heading": heading or "",
            "section_path": section_path or "",
            "section_title": chunk.metadata.get("section_title") or "",
            "section_level": int(chunk.metadata.get("section_level", 0)),
            "chunk_type": chunk.metadata.get("chunk_type", "text"),
            "table_context": chunk.metadata.get("table_context") or "",
            "original_length": int(original_length),
            "truncated": bool(was_truncated),
            "has_table": bool(chunk.metadata.get("has_table", False)),
            "sequential_position": int(chunk.metadata.get("sequential_position", i)),
            "prev_chunk_ids": ",".join(map(str, prev_chunk_ids)) if prev_chunk_ids else "",
            "next_chunk_ids": ",".join(map(str, next_chunk_ids)) if next_chunk_ids else "",
            "start_line": int(chunk.metadata.get("start_line", 0)),
        }
        
        # Final verification
        if len(content) > EMBEDDING_MAX_CHARS:
            content = content[:EMBEDDING_MAX_CHARS] + "..."
            was_truncated = True
        
        final_tokens = token_tracker.count_tokens(content)
        if final_tokens > EMBEDDING_MAX_TOKENS:
            content = content[:int(EMBEDDING_MAX_CHARS * 0.8)] + "..."
            was_truncated = True
        
        # Create document for vector store
        doc = Document(
            page_content=content,
            metadata=chroma_metadata
        )
        
        # Add to vector store
        try:
            vector_store.add_documents([doc])
        except Exception as e:
            logger.error(f"Failed to add chunk {vector_number}: {e}")
            content = content[:int(EMBEDDING_MAX_CHARS * 0.5)] + "..."
            doc.page_content = content
            try:
                vector_store.add_documents([doc])
            except Exception as e2:
                logger.error(f"Still failed, skipping chunk {vector_number}")
                continue
        
        # Add chunk node to graph
        chunk_node = document_graph.add_chunk_node(vector_number, doc)
        
        # Add edges: chunk belongs_to section
        if section_path:
            section_node = document_graph.section_nodes.get(section_path)
            if section_node:
                document_graph.add_edge(section_node, chunk_node, relation="contains")
                document_graph.add_edge(chunk_node, section_node, relation="belongs_to")
        
        # Add edges: chunk follows previous/next chunks
        for prev_id in prev_chunk_ids:
            prev_node = document_graph.chunk_nodes.get(prev_id)
            if prev_node:
                document_graph.add_edge(prev_node, chunk_node, relation="follows")
        
        for next_id in next_chunk_ids:
            next_node = document_graph.chunk_nodes.get(next_id)
            if next_node:
                document_graph.add_edge(chunk_node, next_node, relation="follows")
        
        # Create full metadata for JSON
        full_metadata = chroma_metadata.copy()
        full_metadata["prev_chunk_ids"] = prev_chunk_ids
        full_metadata["next_chunk_ids"] = next_chunk_ids
        
        chunk_detail = {
            "vector_number": vector_number,
            "heading": heading,
            "section_path": section_path,
            "content": content,
            "content_length": len(content),
            "metadata": full_metadata
        }
        
        json_mapping.append(chunk_detail)
        processed_chunks.append(chunk_detail)
        token_tracker.stats["total_chunks"] += 1

        if entity_extractor:
            chunk_identifier = f"{doc_stem}::chunk::{vector_number}"
            extraction_result: ExtractionResult = entity_extractor.extract(
                chunk.page_content,
                chunk_id=chunk_identifier,
                extra_context={
                    "section_path": section_path,
                    "heading": heading,
                },
            )

            if extraction_result.entities or extraction_result.relations:
                knowledge_records.append(
                    {
                        "chunk_id": chunk_identifier,
                        "heading": heading,
                        "section_path": section_path,
                        "entities": [
                            {
                                "name": entity.name,
                                "type": entity.type,
                                "description": entity.description,
                                "aliases": entity.aliases,
                                "source_ids": entity.source_ids,
                            }
                            for entity in extraction_result.entities
                        ],
                        "relations": [
                            {
                                "source": relation.source,
                                "relation": relation.relation,
                                "target": relation.target,
                                "evidence": relation.evidence,
                            }
                            for relation in extraction_result.relations
                        ],
                        "raw_response": extraction_result.raw_response,
                    }
                )
    
    # Compute similarity edges between chunks
    logger.info("Computing similarity edges between chunks...")
    similarity_threshold = 0.50  # Similarity threshold for L2 distance (higher = more similar)
    max_similar_per_chunk = 5  # Maximum similar chunks to connect per chunk
    
    similarity_edges_added = 0
    all_chunk_ids = sorted(document_graph.chunk_nodes.keys())
    
    # Debug: Track distance ranges
    distance_samples = []
    
    for i, chunk_id in enumerate(all_chunk_ids):
        if (i + 1) % 50 == 0:
            logger.info(f"  Computing similarities for chunk {i + 1}/{len(all_chunk_ids)}...")
        
        chunk_node = document_graph.chunk_nodes.get(chunk_id)
        if not chunk_node:
            continue
        
        # Get the chunk content from vector store
        try:
            # Use the chunk's content to find similar chunks
            chunk_doc = None
            for chunk_detail in processed_chunks:
                if chunk_detail.get("vector_number") == chunk_id:
                    chunk_doc = Document(
                        page_content=chunk_detail.get("content", ""),
                        metadata=chunk_detail.get("metadata", {})
                    )
                    break
            
            if not chunk_doc or not chunk_doc.page_content.strip():
                continue
            
            # Find similar chunks using vector similarity search
            # Search for more than we need to account for the chunk itself
            similar_results = vector_store.similarity_search_with_score(
                chunk_doc.page_content,
                k=max_similar_per_chunk + 3
            )
            
            if not similar_results:
                if i < 3:  # Debug for first few
                    logger.warning(f"  No results for chunk {chunk_id}")
                continue
            
            if i < 3:  # Debug for first few
                logger.info(f"  Chunk {chunk_id}: Found {len(similar_results)} search results")
            
            similar_count = 0
            skipped_same = 0
            skipped_none = 0
            skipped_not_in_graph = 0
            
            for similar_doc, distance_score in similar_results:
                similar_chunk_id = similar_doc.metadata.get("chunk_index")
                
                # Debug for first chunk
                if i == 0 and similar_count < 5:
                    logger.info(f"    Result: chunk_id={similar_chunk_id}, distance={distance_score:.4f}, metadata={similar_doc.metadata.get('chunk_index')}")
                
                # Skip if same chunk, invalid ID, or already connected
                if similar_chunk_id is None:
                    skipped_none += 1
                    if i < 2:
                        logger.info(f"    Skipped: chunk_id is None")
                    continue
                if similar_chunk_id == chunk_id:
                    skipped_same += 1
                    continue
                if similar_chunk_id not in document_graph.chunk_nodes:
                    skipped_not_in_graph += 1
                    if i < 2:
                        logger.info(f"    Skipped: chunk {similar_chunk_id} not in graph (graph has {len(document_graph.chunk_nodes)} chunks)")
                    continue
                
                # Convert distance to similarity
                # Chroma is returning L2/Euclidean distance (not cosine distance!)
                # L2 distance can be any positive number, where:
                # - 0 = identical
                # - Lower values = more similar
                # - Higher values = less similar
                # 
                # For L2 distance, we need to convert to similarity using:
                # similarity = 1 / (1 + normalized_distance)
                # or exponential decay: similarity = exp(-distance/scale)
                
                # First, normalize the distance by finding a reasonable scale
                # Based on the debug output, distances are typically 200-400 range
                # For identical chunks, distance should be close to 0
                # For very different chunks, distance can be 300-400+
                
                # Use inverse relationship with normalization
                # Scale factor: use a value that makes sense for the distance range
                # If distance is 0, similarity should be 1.0
                # If distance is ~300 (typical), similarity should be low
                # If distance is ~100 (very similar), similarity should be higher
                
                # Method 1: Inverse with scale (works well for L2 distance)
                # Scale of 150 means: 
                #   distance 0 -> similarity 1.0 (identical)
                #   distance 100 -> similarity 0.6 (very similar)
                #   distance 200 -> similarity 0.57 (similar)
                #   distance 300 -> similarity 0.5 (moderately similar)
                #   distance 400 -> similarity 0.43 (less similar)
                scale_factor = 150.0
                similarity = 1.0 / (1.0 + (distance_score / scale_factor))
                
                # Clamp to [0, 1] range
                similarity = max(0.0, min(1.0, similarity))
                
                # Debug for first chunk
                if i == 0 and similar_count < 3:
                    logger.info(f"    Distance={distance_score:.4f} -> Similarity={similarity:.4f} (threshold={similarity_threshold})")
                
                # Clamp similarity to [0, 1] range
                similarity = max(0.0, min(1.0, similarity))
                
                # Collect distance samples for debugging
                if len(distance_samples) < 20:
                    distance_samples.append((distance_score, similarity))
                
                # Debug logging for first few chunks
                if i < 3 and similar_count < 2:
                    logger.info(f"  Chunk {chunk_id} -> {similar_chunk_id}: distance={distance_score:.4f}, similarity={similarity:.4f}, threshold={similarity_threshold}")
                
                # Only add if similarity is above threshold
                if similarity >= similarity_threshold:
                    similar_node = document_graph.chunk_nodes.get(similar_chunk_id)
                    if similar_node:
                        # Check if "follows" edge already exists (don't add similarity if chunks are adjacent)
                        existing_edge_1 = document_graph.graph.get_edge_data(chunk_node, similar_node)
                        existing_edge_2 = document_graph.graph.get_edge_data(similar_node, chunk_node)
                        
                        # Skip if "follows" edge exists in either direction
                        has_follows_edge = False
                        if existing_edge_1 and existing_edge_1.get("relation") == "follows":
                            has_follows_edge = True
                        if existing_edge_2 and existing_edge_2.get("relation") == "follows":
                            has_follows_edge = True
                        
                        if has_follows_edge:
                            if i < 3:
                                logger.debug(f"    Skipping similarity edge {chunk_id} <-> {similar_chunk_id}: already connected via 'follows'")
                            continue
                        
                        # Check if similarity edge already exists (bidirectional check)
                        if (existing_edge_1 and existing_edge_1.get("relation") == "similar_to") or \
                           (existing_edge_2 and existing_edge_2.get("relation") == "similar_to"):
                            continue  # Edge already exists
                        
                        # Add bidirectional similarity edge
                        document_graph.add_edge(
                            chunk_node, 
                            similar_node, 
                            relation="similar_to",
                            similarity=similarity
                        )
                        similarity_edges_added += 1
                        similar_count += 1
                        
                        if similar_count >= max_similar_per_chunk:
                            break
            
            # Debug summary for first few chunks
            if i < 3:
                logger.info(f"  Chunk {chunk_id} summary: {len(similar_results)} results, skipped: same={skipped_same}, none={skipped_none}, not_in_graph={skipped_not_in_graph}, added={similar_count}")
                
        except Exception as e:
            logger.warning(f"Error computing similarity for chunk {chunk_id}: {e}", exc_info=True)
            continue
    
    # Log distance statistics for debugging
    if distance_samples:
        distances = [d[0] for d in distance_samples]
        similarities = [d[1] for d in distance_samples]
        logger.info(f"Distance sample stats: min={min(distances):.4f}, max={max(distances):.4f}, avg={sum(distances)/len(distances):.4f}")
        logger.info(f"Similarity sample stats: min={min(similarities):.4f}, max={max(similarities):.4f}, avg={sum(similarities)/len(similarities):.4f}")
        logger.info(f"Threshold: {similarity_threshold}, Edges above threshold: {sum(1 for s in similarities if s >= similarity_threshold)}")
    
    logger.info(f"Added {similarity_edges_added} similarity edges between chunks")
    
    # Save files
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_mapping, f, indent=2, ensure_ascii=False)
    
    document_graph.save(graph_output_path)
    
    state['processed_chunks'] = processed_chunks
    state['vector_store'] = vector_store
    state['document_graph'] = document_graph
    state['json_mapping'] = json_mapping

    if entity_extractor and knowledge_records:
        knowledge_output_dir = document_output_dir / "knowledge"
        knowledge_output_dir.mkdir(parents=True, exist_ok=True)
        knowledge_path = knowledge_output_dir / f"{doc_stem}_chunk_knowledge.json"
        knowledge_graph = graph_builder.build_graph(doc_stem, knowledge_records)
        graph_builder_path = knowledge_output_dir / f"{doc_stem}_knowledge_graph.json"
        graph_builder.save_graph(knowledge_graph, graph_builder_path)
        knowledge_payload = {
            "document_id": doc_stem,
            "chunks": knowledge_records,
            "entities": graph_builder.aggregate_entities(knowledge_graph),
            "relations": graph_builder.aggregate_relations(knowledge_graph),
        }
        with open(knowledge_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_payload, f, indent=2, ensure_ascii=False)
        logger.info(
            "Stored knowledge extraction for %d chunks to %s",
            len(knowledge_records),
            knowledge_path,
        )
        logger.info("Knowledge graph saved to %s", graph_builder_path)
    
    # Log stats
    final_stats = token_tracker.get_stats()
    state['token_usage'] = final_stats
    logger.info("=" * 60)
    logger.info("Token Usage Statistics:")
    logger.info(f"  Total chunks processed: {final_stats['total_chunks']}")
    logger.info(f"  Total embedding tokens: {final_stats['embedding_tokens']}")
    logger.info(f"  Truncated chunks: {final_stats['truncated_chunks']}")
    logger.info("=" * 60)
    
    logger.info(f"Completed processing all {len(processed_chunks)} chunks")
    logger.info(f"Vector store saved to: {vector_db_dir}")
    logger.info(f"JSON mapping saved to: {json_output_path}")
    logger.info(f"Document graph saved to: {graph_output_path}")
    logger.info(f"Graph: {len(document_graph.graph.nodes)} nodes, {len(document_graph.graph.edges)} edges")
    
    return state

# ---------------- LANGGRAPH WORKFLOW ---------------- 
def create_vectorization_workflow() -> StateGraph:
    """Create LangGraph workflow for vectorization"""
    workflow = StateGraph(VectorizerState)
    
    workflow.add_node("load_markdown", load_markdown)
    workflow.add_node("process_chunks", process_chunks_one_by_one)
    
    workflow.set_entry_point("load_markdown")
    workflow.add_edge("load_markdown", "process_chunks")
    workflow.add_edge("process_chunks", END)
    
    return workflow.compile()


def vectorize_markdown_content(
    markdown_content: str,
    output_root: Path,
    doc_stem: str,
) -> Dict[str, Any]:
    """
    Vectorize markdown string (e.g. from email) without loading from file.
    Writes to output_root / doc_stem / (vector_db, vector_mapping, document_graph, knowledge).
    Returns final state dict.
    """
    logger.info("Vectorizing markdown content for doc_stem=%s", doc_stem)
    chunks, structure = parse_markdown_enhanced(markdown_content)
    if not chunks:
        logger.warning("No chunks produced for %s", doc_stem)
        return {}

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    document_output_dir = output_root / doc_stem
    document_output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = document_output_dir / f"{doc_stem}.md"

    initial_state: VectorizerState = {
        "markdown_file": str(synthetic_path),
        "document_path": str(synthetic_path),
        "chunks": chunks,
        "structure": structure,
        "processed_chunks": [],
        "vector_store": None,
        "document_graph": DocumentGraph(),
        "json_mapping": [],
        "output_folder": str(document_output_dir),
        "output_root": str(output_root),
    }

    final_state = process_chunks_one_by_one(initial_state)
    return final_state


# ---------------- HELPER FUNCTIONS ---------------- 
def check_ollama_running() -> Tuple[bool, List[str]]:
    """Check if Ollama server is running"""
    try:
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
            response = urllib.request.urlopen(req, timeout=5)
            data = json.loads(response.read().decode())
            models = [model.get('name', '') for model in data.get('models', [])]
            return True, models
        except urllib.error.URLError:
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                response = urllib.request.urlopen(req, timeout=5)
                data = json.loads(response.read().decode())
                models = [model.get('name', '') for model in data.get('models', [])]
                return True, models
            except:
                return False, []
    except Exception as e:
        logger.debug(f"Ollama check error: {e}")
        return False, []
# ---------------- MAIN FUNCTION ---------------- 
def main():
    """Main entry point"""
    logger.info("Checking Ollama server connection...")
    is_running, available_models = check_ollama_running()
    if not is_running:
        logger.error("=" * 60)
        logger.error("Ollama server is not running!")
        logger.error("=" * 60)
        logger.error("Please start Ollama server:")
        logger.error("  1. Open a new terminal")
        logger.error("  2. Run: ollama serve")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info("✓ Ollama server is running")
    
    if TIKTOKEN_AVAILABLE:
        logger.info("✓ Token tracking enabled with tiktoken")
    else:
        logger.warning("⚠ tiktoken not available - using fallback token counting")
    
    if not NETWORKX_AVAILABLE:
        logger.error("=" * 60)
        logger.error("networkx is required for Plan E!")
        logger.error("=" * 60)
        logger.error("Install with: pip install networkx")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info("✓ NetworkX available for graph structure")
    
    # Accept .md file or folder containing .md as argument
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        logger.error("Please provide a .md file path (or folder containing .md): python vectorizerE.py <path_to_file.md>")
        sys.exit(1)
    
    if not input_path.exists():
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("VectorizerE - Graph-Based Markdown Vectorizer")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Token tracking: {'Enabled' if TIKTOKEN_AVAILABLE else 'Fallback'}")
    logger.info("=" * 60)

    md_files_in_root = list(input_path.glob("*.md")) if input_path.is_dir() else []
    subfolders = [d for d in input_path.iterdir() if d.is_dir()] if input_path.is_dir() else []

    # Helper to run workflow on a single .md file or folder
    def run_for_doc(doc_path: Path):
        if doc_path.is_file() and doc_path.suffix.lower() == ".md":
            md_path = doc_path
            output_folder = doc_path.parent
        elif doc_path.is_dir():
            md_files = list(doc_path.glob("*.md"))
            if not md_files:
                logger.warning(f"No .md file found in folder: {doc_path}, skipping")
                return
            md_path = md_files[0]
            output_folder = doc_path
        else:
            logger.warning(f"Invalid path: {doc_path}, skipping")
            return

        output_root = Path("output")
        logger.info("=" * 60)
        logger.info(f"Processing: {md_path.name}")
        logger.info(f"Output folder: {output_root / md_path.stem}")
        logger.info("=" * 60)

        initial_state: VectorizerState = {
            "markdown_file": str(md_path if md_path.is_file() else doc_path),
            "document_path": str(md_path),
            "chunks": [],
            "structure": {},
            "processed_chunks": [],
            "vector_store": None,
            "document_graph": DocumentGraph(),
            "json_mapping": [],
            "output_folder": str(output_folder),
            "output_root": str(output_root)
        }

        workflow = create_vectorization_workflow()
        final_state = workflow.invoke(initial_state)

        logger.info("=" * 60)
        logger.info("Vectorization Complete!")
        logger.info("=" * 60)
        logger.info(f"Total vectors created: {len(final_state['json_mapping'])}")
        doc_stem = md_path.stem
        document_output_dir = Path(final_state.get("document_output_dir", output_root / doc_stem))
        logger.info(f"Output folder: {document_output_dir}")
        logger.info(f"Vector DB: {document_output_dir / 'vector_db' / doc_stem}")
        logger.info(f"JSON mapping: {document_output_dir / f'{doc_stem}_vector_mapping.json'}")
        logger.info(f"Document graph: {document_output_dir / f'{doc_stem}_document_graph.json'}")
        logger.info("=" * 60)
        logger.info("")

    try:
        # Mode 1: input is a .md file directly
        if input_path.is_file() and input_path.suffix.lower() == ".md":
            run_for_doc(input_path)
        # Mode 2: input is folder containing .md
        elif md_files_in_root:
            run_for_doc(input_path)
        # Mode 3: input is folder with subfolders, each containing .md
        elif subfolders:
            logger.info(f"Detected {len(subfolders)} subfolders under {input_path}")
            for doc_folder in sorted(subfolders):
                run_for_doc(doc_folder)
            logger.info("All document folders processed.")
        else:
            logger.error(f"No .md file found. Please provide a .md file or folder containing .md")
            sys.exit(1)

        logger.info("Graph-based retrieval is now available for processed documents!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during vectorization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
