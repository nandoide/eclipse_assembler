#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECLIPSE MULTI-PROJECT MANAGER & RESOLUTION SYSTEM
═══════════════════════════════════════════════════════════════════════════════
Discovers, resolves, and manages project workspaces in '010_in/' and '040_out/'.

Project Directory Naming Convention in 010_in/:
  <project_name>_<film_style>[_preproc]

Examples:
  - 010_in/solar26v1_standard     -> film_style: standard, preproc: False
  - 010_in/solar26v2_art_preproc  -> film_style: art,      preproc: True

Resolution Rules:
  1. If --project is specified: matches exact folder or folder prefix in 010_in/.
  2. If --project is NOT specified: automatically defaults to the most recently
     modified project directory in 010_in/.
  3. Preprocessing mode is determined directly by the presence of 'preproc' (or 'prep')
     in the project directory name.
  4. Corresponding output directory is created at 040_out/<project_name>/.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from typing import List, Dict, Optional


def get_repo_root(anchor_file: Optional[str] = None) -> str:
    """Returns absolute path to the git repository root."""
    if anchor_file:
        return os.path.dirname(os.path.dirname(os.path.abspath(anchor_file)))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_available_projects(repo_root: Optional[str] = None, in_base: str = "010_in") -> List[Dict]:
    """
    Scans in_base directory for project folders and returns a list sorted by
    most recent modification time first.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    in_base_abs = in_base if os.path.isabs(in_base) else os.path.join(repo_root, in_base)
    if not os.path.exists(in_base_abs):
        return []

    projects = []
    for item in os.listdir(in_base_abs):
        if item.startswith(".") or item.startswith("tmp_"):
            continue
        item_path = os.path.join(in_base_abs, item)
        if os.path.isdir(item_path):
            try:
                mtime = os.path.getmtime(item_path)
            except OSError:
                mtime = 0.0

            name_lower = item.lower()
            is_preprocessed = ("preproc" in name_lower or "prep" in name_lower)

            if "art" in name_lower:
                film_style = "art"
            elif "standard" in name_lower:
                film_style = "standard"
            else:
                film_style = "standard"

            projects.append({
                "name": item,
                "path": item_path,
                "mtime": mtime,
                "film_style": film_style,
                "is_preprocessed": is_preprocessed,
            })

    # Sort descending by modification time
    projects.sort(key=lambda p: p["mtime"], reverse=True)
    return projects


def get_default_project(repo_root: Optional[str] = None, in_base: str = "010_in") -> Optional[Dict]:
    """Returns the most recently modified project in 010_in/."""
    projects = list_available_projects(repo_root=repo_root, in_base=in_base)
    if projects:
        return projects[0]
    return None


def resolve_project(
    project_arg: Optional[str] = None,
    repo_root: Optional[str] = None,
    in_base: str = "010_in",
    out_base: str = "040_out"
) -> Dict:
    """
    Resolves project configuration.
    If project_arg is given: matches exact name, prefix, or path.
    If project_arg is None: selects the most recently modified project directory.
    If no subdirectories exist in 010_in: falls back to 010_in/ root.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    in_base_abs = in_base if os.path.isabs(in_base) else os.path.join(repo_root, in_base)
    out_base_abs = out_base if os.path.isabs(out_base) else os.path.join(repo_root, out_base)

    root_in = os.path.join(repo_root, "010_in")

    # If in_base_abs is already a project subfolder (parent is 010_in) and no projects inside it:
    if os.path.basename(os.path.dirname(in_base_abs)) == "010_in" and os.path.isdir(in_base_abs):
        folder_name = os.path.basename(in_base_abs)
        name_lower = folder_name.lower()
        return {
            "name": folder_name,
            "in_dir": in_base_abs,
            "out_dir": os.path.join(out_base_abs, folder_name) if not out_base_abs.endswith(folder_name) else out_base_abs,
            "film_style": "art" if "art" in name_lower else "standard",
            "is_preprocessed": ("preproc" in name_lower or "prep" in name_lower),
            "mtime": os.path.getmtime(in_base_abs),
        }

    search_container = in_base_abs if in_base_abs == root_in or not os.path.exists(root_in) else root_in
    projects = list_available_projects(repo_root=repo_root, in_base=search_container)

    selected_project = None
    if project_arg:
        # Strip trailing slashes
        clean_arg = os.path.normpath(project_arg).strip("/\\")
        base_arg = os.path.basename(clean_arg)

        # 1. Exact match
        for p in projects:
            if p["name"] == base_arg or p["name"] == clean_arg:
                selected_project = p
                break

        # 2. Prefix match (e.g. 'solar26v1' matches 'solar26v1_standard')
        if not selected_project:
            for p in projects:
                if p["name"].startswith(base_arg) or p["name"].startswith(clean_arg):
                    selected_project = p
                    break

        # 3. Direct directory path
        if not selected_project and os.path.isdir(clean_arg):
            name = os.path.basename(clean_arg)
            name_lower = name.lower()
            selected_project = {
                "name": name,
                "path": os.path.abspath(clean_arg),
                "mtime": os.path.getmtime(clean_arg),
                "film_style": "art" if "art" in name_lower else "standard",
                "is_preprocessed": ("preproc" in name_lower or "prep" in name_lower),
            }

        if not selected_project:
            available_names = [p["name"] for p in projects]
            raise ValueError(
                f"Project '{project_arg}' not found in '{in_base}'. Available projects: {available_names}"
            )
    else:
        # Default: most recent project
        if projects:
            selected_project = projects[0]
        else:
            # Fallback for flat 010_in/ directory
            name_lower = os.path.basename(in_base_abs).lower()
            selected_project = {
                "name": "default",
                "path": in_base_abs,
                "mtime": os.path.getmtime(in_base_abs) if os.path.exists(in_base_abs) else 0.0,
                "film_style": "standard",
                "is_preprocessed": ("preproc" in name_lower or "prep" in name_lower),
            }

    proj_name = selected_project["name"]
    proj_in_dir = selected_project["path"]
    if proj_name == "default":
        proj_out_dir = out_base_abs
    else:
        proj_out_dir = os.path.join(out_base_abs, proj_name)

    os.makedirs(proj_out_dir, exist_ok=True)

    return {
        "name": proj_name,
        "in_dir": proj_in_dir,
        "out_dir": proj_out_dir,
        "film_style": selected_project["film_style"],
        "is_preprocessed": selected_project["is_preprocessed"],
        "mtime": selected_project["mtime"],
    }


def add_project_argument(parser, default=None, help_text="Project folder name in 010_in/ (default: most recently modified)"):
    """Convenience helper to add --project / -p argument to an argparse.ArgumentParser."""
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=default,
        help=help_text
    )


if __name__ == "__main__":
    print("=================================================================")
    print("ECLIPSE MULTI-PROJECT MANAGER")
    print("=================================================================")
    projs = list_available_projects()
    print(f"Discovered {len(projs)} projects in 010_in/:")
    for i, p in enumerate(projs, 1):
        flag = " [DEFAULT: MOST RECENT]" if i == 1 else ""
        print(f"  {i}. {p['name']}{flag}")
        print(f"     • In Dir         : {p['path']}")
        print(f"     • Default Style  : {p['film_style'].upper()}")
        print(f"     • Preprocessing  : {'ENABLED' if p['is_preprocessed'] else 'DISABLED'}")

    active = resolve_project()
    print("\nActive Resolved Project:")
    print(f"  • Name            : {active['name']}")
    print(f"  • In Directory    : {active['in_dir']}")
    print(f"  • Out Directory   : {active['out_dir']}")
    print(f"  • Film Style      : {active['film_style']}")
    print(f"  • Is Preprocessed : {active['is_preprocessed']}")
    print("=================================================================")
