"""
Antimatters Core Services
=========================
Persistent storage and management for:
- Workspaces (project contexts)
- Conversations (chat history with full message replay)
- Knowledge/Evolution (extracted learnings, artifacts, feedback)

Inspired by:
- Antigravity's progressive disclosure and knowledge management
- ADK's Session/State/Memory architecture
- AG-UI's artifact and state management
"""

import json
import os
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib


# =============================================================================
# Database Setup
# =============================================================================

DB_PATH = Path(__file__).parent.parent / "data" / "antimatters.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db():
    """Get database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_db()
    cursor = conn.cursor()

    # Workspaces table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    """)

    # Messages table (full conversation history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)

    # Artifacts table (replaces file-based storage)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            workspace_id TEXT,
            type TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    """)

    # Knowledge/Learnings table (Evolution)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source_conversation_id TEXT,
            source_artifact_id TEXT,
            tags TEXT DEFAULT '[]',
            embedding TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    """)

    # Feedback/Comments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            conversation_id TEXT,
            user_id TEXT DEFAULT 'scientist',
            comment TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
        )
    """)

    # Create default workspace if none exists
    cursor.execute("SELECT COUNT(*) FROM workspaces")
    if cursor.fetchone()[0] == 0:
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO workspaces (id, name, description, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "ws_core",
            "core",
            "Default Antimatters workspace for IDP ensemble docking research",
            json.dumps({
                "binding_site": [125, 133, 136],
                "default_exhaustiveness": 32,
                "target_protein": "alpha-synuclein"
            }),
            now, now
        ))

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


# Initialize on import
init_db()


# =============================================================================
# Workspace Service
# =============================================================================

class WorkspaceService:
    """Manages workspaces - project contexts with configuration."""

    def create(self, name: str, description: str = "", config: Dict = None) -> Dict:
        """Create a new workspace."""
        workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO workspaces (id, name, description, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (workspace_id, name, description, json.dumps(config or {}), now, now))
        conn.commit()
        conn.close()

        return self.get(workspace_id)

    def get(self, workspace_id: str) -> Optional[Dict]:
        """Get workspace by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "config": json.loads(row["config"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None

    def list_all(self) -> List[Dict]:
        """List all workspaces."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workspaces ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "config": json.loads(row["config"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        } for row in rows]

    def update(self, workspace_id: str, **kwargs) -> Optional[Dict]:
        """Update workspace fields."""
        conn = get_db()
        cursor = conn.cursor()

        updates = []
        values = []
        for key in ["name", "description"]:
            if key in kwargs:
                updates.append(f"{key} = ?")
                values.append(kwargs[key])

        if "config" in kwargs:
            updates.append("config = ?")
            values.append(json.dumps(kwargs["config"]))

        if updates:
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(workspace_id)

            cursor.execute(f"""
                UPDATE workspaces SET {', '.join(updates)} WHERE id = ?
            """, values)
            conn.commit()

        conn.close()
        return self.get(workspace_id)

    def delete(self, workspace_id: str) -> bool:
        """Delete a workspace (and all associated data)."""
        conn = get_db()
        cursor = conn.cursor()

        # Delete associated conversations and messages
        cursor.execute("""
            DELETE FROM messages WHERE conversation_id IN
            (SELECT id FROM conversations WHERE workspace_id = ?)
        """, (workspace_id,))
        cursor.execute("DELETE FROM conversations WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM artifacts WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM knowledge WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))

        conn.commit()
        conn.close()
        return True

    def get_stats(self, workspace_id: str) -> Dict:
        """Get workspace statistics."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations WHERE workspace_id = ?", (workspace_id,))
        conv_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM artifacts WHERE workspace_id = ?", (workspace_id,))
        artifact_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge WHERE workspace_id = ?", (workspace_id,))
        knowledge_count = cursor.fetchone()[0]

        conn.close()

        return {
            "conversations": conv_count,
            "artifacts": artifact_count,
            "knowledge_items": knowledge_count
        }


# =============================================================================
# Conversation Service
# =============================================================================

class ConversationService:
    """Manages conversations with full message history."""

    def create(self, workspace_id: str, title: str = None, initial_message: str = None) -> Dict:
        """Create a new conversation."""
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Generate title from initial message if not provided
        if not title and initial_message:
            title = initial_message[:50] + ("..." if len(initial_message) > 50 else "")
        title = title or "New Simulation"

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (id, workspace_id, title, status, created_at, updated_at)
            VALUES (?, ?, ?, 'active', ?, ?)
        """, (conv_id, workspace_id, title, now, now))

        # Add initial message if provided
        if initial_message:
            msg_id = f"msg_{uuid.uuid4().hex[:8]}"
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES (?, ?, 'user', ?, ?)
            """, (msg_id, conv_id, initial_message, now))

        conn.commit()
        conn.close()

        return self.get(conv_id)

    def get(self, conversation_id: str, include_messages: bool = False) -> Optional[Dict]:
        """Get conversation by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        conv = {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "title": row["title"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

        if include_messages:
            cursor.execute("""
                SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC
            """, (conversation_id,))
            conv["messages"] = [{
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "tool_calls": json.loads(m["tool_calls"]) if m["tool_calls"] else None,
                "metadata": json.loads(m["metadata"]),
                "created_at": m["created_at"]
            } for m in cursor.fetchall()]

        conn.close()
        return conv

    def list_by_workspace(self, workspace_id: str, limit: int = 50) -> List[Dict]:
        """List conversations for a workspace."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*,
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
            FROM conversations c
            WHERE workspace_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (workspace_id, limit))
        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "title": row["title"],
            "status": row["status"],
            "message_count": row["message_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        } for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str,
                    tool_calls: List[Dict] = None, metadata: Dict = None) -> Dict:
        """Add a message to a conversation."""
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO messages (id, conversation_id, role, content, tool_calls, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            msg_id, conversation_id, role, content,
            json.dumps(tool_calls) if tool_calls else None,
            json.dumps(metadata or {}),
            now
        ))

        # Update conversation timestamp
        cursor.execute("""
            UPDATE conversations SET updated_at = ? WHERE id = ?
        """, (now, conversation_id))

        conn.commit()
        conn.close()

        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "metadata": metadata or {},
            "created_at": now
        }

    def update_title(self, conversation_id: str, title: str) -> Optional[Dict]:
        """Update conversation title."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?
        """, (title, datetime.now().isoformat(), conversation_id))
        conn.commit()
        conn.close()
        return self.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
        return True


# =============================================================================
# Artifact Service (Enhanced)
# =============================================================================

class ArtifactService:
    """Manages artifacts with database persistence and search."""

    def create(self, artifact_type: str, content: Any, conversation_id: str = None,
               workspace_id: str = None, title: str = None, metadata: Dict = None) -> Dict:
        """Create a new artifact."""
        artifact_id = f"artifact_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Auto-generate title if not provided
        if not title:
            title = self._generate_title(artifact_type, content)

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO artifacts (id, conversation_id, workspace_id, type, title, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            artifact_id, conversation_id, workspace_id, artifact_type,
            title, json.dumps(content), json.dumps(metadata or {}), now
        ))
        conn.commit()
        conn.close()

        return self.get(artifact_id)

    def get(self, artifact_id: str) -> Optional[Dict]:
        """Get artifact by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "workspace_id": row["workspace_id"],
                "type": row["type"],
                "title": row["title"],
                "content": json.loads(row["content"]),
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"]
            }
        return None

    def list_by_workspace(self, workspace_id: str, artifact_type: str = None, limit: int = 100) -> List[Dict]:
        """List artifacts for a workspace."""
        conn = get_db()
        cursor = conn.cursor()

        if artifact_type:
            cursor.execute("""
                SELECT * FROM artifacts WHERE workspace_id = ? AND type = ?
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, artifact_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM artifacts WHERE workspace_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "workspace_id": row["workspace_id"],
            "type": row["type"],
            "title": row["title"],
            "content": json.loads(row["content"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"]
        } for row in rows]

    def list_by_conversation(self, conversation_id: str) -> List[Dict]:
        """List artifacts for a specific conversation."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM artifacts WHERE conversation_id = ?
            ORDER BY created_at ASC
        """, (conversation_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "workspace_id": row["workspace_id"],
            "type": row["type"],
            "title": row["title"],
            "content": json.loads(row["content"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"]
        } for row in rows]

    def update(self, artifact_id: str, content: Any = None, title: str = None,
               metadata: Dict = None) -> Optional[Dict]:
        """Update an artifact."""
        conn = get_db()
        cursor = conn.cursor()

        updates = []
        values = []

        if content is not None:
            updates.append("content = ?")
            values.append(json.dumps(content))
        if title is not None:
            updates.append("title = ?")
            values.append(title)
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(json.dumps(metadata))

        if updates:
            values.append(artifact_id)
            cursor.execute(f"""
                UPDATE artifacts SET {', '.join(updates)} WHERE id = ?
            """, values)
            conn.commit()

        conn.close()
        return self.get(artifact_id)

    def search(self, query: str, workspace_id: str = None, limit: int = 20) -> List[Dict]:
        """Search artifacts by title or content."""
        conn = get_db()
        cursor = conn.cursor()

        search_pattern = f"%{query}%"

        if workspace_id:
            cursor.execute("""
                SELECT * FROM artifacts
                WHERE workspace_id = ? AND (title LIKE ? OR content LIKE ?)
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, search_pattern, search_pattern, limit))
        else:
            cursor.execute("""
                SELECT * FROM artifacts
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY created_at DESC LIMIT ?
            """, (search_pattern, search_pattern, limit))

        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "content": json.loads(row["content"]),
            "created_at": row["created_at"]
        } for row in rows]

    def _generate_title(self, artifact_type: str, content: Any) -> str:
        """Generate a title based on artifact type and content."""
        if artifact_type == "task_list":
            return "Research Protocol"
        elif artifact_type == "structure_3d":
            ped_id = content.get("metadata", {}).get("ped_id", "Unknown")
            return f"Structure: {ped_id}"
        elif artifact_type == "docking_result":
            return "Docking Analysis"
        elif artifact_type == "cluster_visualization":
            n_clusters = content.get("n_clusters", "?")
            return f"Cluster Analysis ({n_clusters} clusters)"
        elif artifact_type == "interaction_map":
            return "Interaction Analysis"
        elif artifact_type == "literature_result":
            return content.get("title", "Literature Evidence")
        elif artifact_type == "scientific_experiment":
            return content.get("title", "Experiment")
        return f"Artifact ({artifact_type})"


# =============================================================================
# Knowledge Service (Evolution)
# =============================================================================

class KnowledgeService:
    """
    Manages extracted knowledge and learnings.

    This implements Antigravity-style self-improvement where:
    - Agent actions contribute to a knowledge base
    - Past learnings inform future interactions
    - Users can annotate and provide feedback
    """

    # Knowledge types
    INSIGHT = "insight"         # Extracted insight from a conversation
    PROCEDURE = "procedure"     # Learned procedure or workflow
    RESULT = "result"           # Important result worth remembering
    FEEDBACK = "feedback"       # User feedback or correction
    REFERENCE = "reference"     # External reference (paper, documentation)

    def extract_from_conversation(self, conversation_id: str, workspace_id: str) -> List[Dict]:
        """
        Extract knowledge items from a completed conversation.
        This is called after a conversation ends to capture learnings.
        """
        conv_service = ConversationService()
        conv = conv_service.get(conversation_id, include_messages=True)

        if not conv or not conv.get("messages"):
            return []

        extracted = []

        # Extract insights from assistant messages
        for msg in conv["messages"]:
            if msg["role"] == "assistant" and msg["content"]:
                # Look for key insights (simplified extraction)
                content = msg["content"]

                # Extract binding energy results
                if "kcal/mol" in content.lower():
                    extracted.append(self.create(
                        workspace_id=workspace_id,
                        knowledge_type=self.RESULT,
                        title="Binding Energy Result",
                        content=content[:500],
                        source_conversation_id=conversation_id,
                        tags=["docking", "binding_energy"]
                    ))

                # Extract successful procedures
                if "successfully" in content.lower() and ("docked" in content.lower() or "clustered" in content.lower()):
                    extracted.append(self.create(
                        workspace_id=workspace_id,
                        knowledge_type=self.PROCEDURE,
                        title="Successful Workflow",
                        content=content[:500],
                        source_conversation_id=conversation_id,
                        tags=["workflow", "success"]
                    ))

        return extracted

    def create(self, workspace_id: str, knowledge_type: str, title: str, content: str,
               source_conversation_id: str = None, source_artifact_id: str = None,
               tags: List[str] = None) -> Dict:
        """Create a new knowledge item."""
        knowledge_id = f"know_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge (id, workspace_id, type, title, content,
                                   source_conversation_id, source_artifact_id, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            knowledge_id, workspace_id, knowledge_type, title, content,
            source_conversation_id, source_artifact_id,
            json.dumps(tags or []), now
        ))
        conn.commit()
        conn.close()

        return self.get(knowledge_id)

    def get(self, knowledge_id: str) -> Optional[Dict]:
        """Get knowledge item by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row["id"],
                "workspace_id": row["workspace_id"],
                "type": row["type"],
                "title": row["title"],
                "content": row["content"],
                "source_conversation_id": row["source_conversation_id"],
                "source_artifact_id": row["source_artifact_id"],
                "tags": json.loads(row["tags"]),
                "created_at": row["created_at"]
            }
        return None

    def list_by_workspace(self, workspace_id: str, knowledge_type: str = None,
                          limit: int = 50) -> List[Dict]:
        """List knowledge items for a workspace."""
        conn = get_db()
        cursor = conn.cursor()

        if knowledge_type:
            cursor.execute("""
                SELECT * FROM knowledge WHERE workspace_id = ? AND type = ?
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, knowledge_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM knowledge WHERE workspace_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "type": row["type"],
            "title": row["title"],
            "content": row["content"],
            "tags": json.loads(row["tags"]),
            "created_at": row["created_at"]
        } for row in rows]

    def search(self, query: str, workspace_id: str = None, limit: int = 10) -> List[Dict]:
        """Search knowledge base."""
        conn = get_db()
        cursor = conn.cursor()

        search_pattern = f"%{query}%"

        if workspace_id:
            cursor.execute("""
                SELECT * FROM knowledge
                WHERE workspace_id = ? AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)
                ORDER BY created_at DESC LIMIT ?
            """, (workspace_id, search_pattern, search_pattern, search_pattern, limit))
        else:
            cursor.execute("""
                SELECT * FROM knowledge
                WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC LIMIT ?
            """, (search_pattern, search_pattern, search_pattern, limit))

        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
            "tags": json.loads(row["tags"]),
            "created_at": row["created_at"]
        } for row in rows]

    def get_relevant_context(self, query: str, workspace_id: str, limit: int = 5) -> str:
        """
        Get relevant knowledge as context for agent prompts.
        This enables the self-improvement loop - past learnings inform future work.
        """
        relevant = self.search(query, workspace_id, limit)

        if not relevant:
            return ""

        context_parts = ["[Relevant knowledge from past research:]"]
        for item in relevant:
            context_parts.append(f"- {item['title']}: {item['content'][:150]}...")

        return "\n".join(context_parts)


# =============================================================================
# Feedback Service
# =============================================================================

class FeedbackService:
    """Manages feedback/comments on artifacts (Google Docs-style)."""

    def add(self, artifact_id: str, comment: str, conversation_id: str = None,
            user_id: str = "scientist") -> Dict:
        """Add feedback to an artifact."""
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (id, artifact_id, conversation_id, user_id, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (feedback_id, artifact_id, conversation_id, user_id, comment, now))
        conn.commit()
        conn.close()

        return self.get(feedback_id)

    def get(self, feedback_id: str) -> Optional[Dict]:
        """Get feedback by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row["id"],
                "artifact_id": row["artifact_id"],
                "conversation_id": row["conversation_id"],
                "user_id": row["user_id"],
                "comment": row["comment"],
                "resolved": bool(row["resolved"]),
                "created_at": row["created_at"]
            }
        return None

    def list_by_artifact(self, artifact_id: str) -> List[Dict]:
        """List feedback for an artifact."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM feedback WHERE artifact_id = ? ORDER BY created_at ASC
        """, (artifact_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{
            "id": row["id"],
            "user_id": row["user_id"],
            "comment": row["comment"],
            "resolved": bool(row["resolved"]),
            "created_at": row["created_at"]
        } for row in rows]

    def resolve(self, feedback_id: str) -> Optional[Dict]:
        """Mark feedback as resolved."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE feedback SET resolved = 1 WHERE id = ?", (feedback_id,))
        conn.commit()
        conn.close()
        return self.get(feedback_id)


# =============================================================================
# Service Instances (Singletons)
# =============================================================================

workspace_service = WorkspaceService()
conversation_service = ConversationService()
artifact_service = ArtifactService()
knowledge_service = KnowledgeService()
feedback_service = FeedbackService()
