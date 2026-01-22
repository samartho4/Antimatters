"""Base components for Antimatters agents."""
import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Define artifacts directory
CORE_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = CORE_ROOT / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def create_markdown_artifact(name: str, content: str, artifact_type: str = "implementation_plan") -> str:
    """
    Create a markdown artifact that will be displayed to the user.
    
    Args:
        name: The title/name of the artifact
        content: The markdown content
        artifact_type: Type of artifact (implementation_plan, task_list, walkthrough, etc.)
        
    Returns:
        Confirmation message with artifact ID
    """
    artifact_id = f"artifact_{uuid.uuid4().hex[:8]}"
    filename = f"{artifact_id}.md"
    path = ARTIFACTS_DIR / filename
    
    artifact_data = {
        "id": artifact_id,
        "type": artifact_type,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "content_path": str(path),
        "content": content
    }
    
    # Save content to MD file
    with open(path, "w") as f:
        f.write(content)
        
    # Save metadata
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(artifact_data, f, indent=2)
        
    # Return metadata for the runner/server to consume
    return artifact_data

def read_artifact(artifact_id: str) -> str:
    """
    Read the content of an artifact.
    
    Args:
        artifact_id: The ID of the artifact to read
        
    Returns:
        The content of the artifact
    """
    # Try finding by ID first
    for meta_file in ARTIFACTS_DIR.glob("*.json"):
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
                if data.get("id") == artifact_id or artifact_id in meta_file.name:
                    with open(Path(data["content_path"]), "r") as cf:
                        return cf.read()
        except Exception:
            continue
            
    return f"Artifact {artifact_id} not found."

def update_artifact(artifact_id: str, content: str) -> str:
    """
    Update an existing artifact.
    
    Args:
        artifact_id: The ID of the artifact to update
        content: The new content
        
    Returns:
        Confirmation message
    """
    for meta_file in ARTIFACTS_DIR.glob("*.json"):
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
            
            if data.get("id") == artifact_id or artifact_id in meta_file.name:
                path = Path(data["content_path"])
                with open(path, "w") as f:
                    f.write(content)
                return f"Artifact {artifact_id} updated."
                
        except Exception:
            continue
            
    return f"Artifact {artifact_id} not found."
