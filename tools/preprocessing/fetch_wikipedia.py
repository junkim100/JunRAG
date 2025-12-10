#!/usr/bin/env python3
"""
Fetch Wikipedia content and save as markdown files.
Reads wiki_links from test.json and val.json, fetches content, and saves to data/markdown/.

Features:
- Deduplicates URLs across the entire dataset (each unique URL fetched only once)
- Creates symlinks/copies for items that reference the same URL
- No limit on number of wiki links per item

Usage:
    uv run tools/preprocessing/fetch_wikipedia.py --max_workers 20
    uv run tools/preprocessing/fetch_wikipedia.py --dataset test --max_workers 20

Output:
    data/markdown/test_0_0.md, test_0_1.md, ... (item-specific files)
    data/markdown/test_canonical_*.md (canonical files for unique URLs)
    data/markdown/val_0_0.md, val_0_1.md, ...
    data/markdown/val_canonical_*.md
"""

import ast
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import fire
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm


def parse_wiki_links_field(wiki_links_value):
    """Handles the wiki_links field which is sometimes a string representation of a list."""
    if isinstance(wiki_links_value, list):
        return wiki_links_value
    if isinstance(wiki_links_value, str):
        try:
            val = ast.literal_eval(wiki_links_value)
            if isinstance(val, list):
                return val
        except Exception:
            stripped = wiki_links_value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            if (stripped.startswith("'") and stripped.endswith("'")) or (
                stripped.startswith('"') and stripped.endswith('"')
            ):
                stripped = stripped[1:-1]

            urls = []
            for x in stripped.split(","):
                x = x.strip().strip("'\"")
                if x and ("http" in x or "wiki" in x):
                    urls.append(x)
            return urls
    return []


def fetch_wikipedia_content(url: str, max_retries: int = 3) -> str:
    """Fetches Wikipedia content from a URL and converts it to markdown."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ):
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            else:
                return ""
        except Exception:
            return ""

    if not response:
        return ""

    try:
        soup = BeautifulSoup(response.content, "html.parser")

        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            content_div = soup.find("div", {"class": "mw-parser-output"})
        if not content_div:
            content_div = soup.find("body")

        if not content_div:
            return ""

        # Remove navigation boxes and headers
        for element in content_div.find_all(
            ["div", "nav", "aside", "table", "span", "sup"],
            class_=re.compile(
                r"navbox|infobox|reference|citation|mw-editsection|hatnote|vertical-navbox|thumb|metadata|ambox|mbox|sistersitebox|navbox-group|mw-jump-link|mw-heading|toc|mw-normal-catlinks|catlinks"
            ),
        ):
            element.decompose()

        for element in content_div.find_all(id=re.compile(r"nav|toc|infobox|catlinks")):
            element.decompose()

        for table in content_div.find_all("table"):
            if table is None:
                continue
            try:
                table_class = table.get("class", []) if table else []
                if any(
                    "nav" in str(c).lower() or "infobox" in str(c).lower()
                    for c in table_class
                ):
                    table.decompose()
                    continue
                links = table.find_all("a")
                if len(links) > 10:
                    table.decompose()
            except Exception:
                continue

        for element in content_div.find_all(["script", "style"]):
            if element:
                try:
                    element.decompose()
                except Exception:
                    pass

        for link in list(content_div.find_all("a")):
            if link:
                try:
                    link_text = link.get_text()
                    link.replace_with(link_text)
                except Exception:
                    pass

        for img in list(content_div.find_all("img")):
            if img:
                try:
                    alt_text = img.get("alt", "") if img else ""
                    placeholder = f"[Image: {alt_text}]" if alt_text else "[Image]"
                    img.replace_with(placeholder)
                except Exception:
                    pass

        unwanted_sections = [
            "See also",
            "Notes",
            "References",
            "Further reading",
            "External links",
            "Bibliography",
            "Citations",
        ]

        headings = content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        for heading in reversed(headings):
            heading_text = heading.get_text(strip=True).lower()
            for unwanted in unwanted_sections:
                if unwanted.lower() in heading_text:
                    heading_level = int(heading.name[1]) if heading.name else 6
                    current = heading
                    elements_to_remove = [current]

                    for sibling in current.next_siblings:
                        if sibling.name and sibling.name.startswith("h"):
                            sibling_level = int(sibling.name[1]) if sibling.name else 6
                            if sibling_level <= heading_level:
                                break
                        elements_to_remove.append(sibling)

                    for elem in elements_to_remove:
                        if elem and elem.name:
                            elem.decompose()
                    break

        markdown_content = md(str(content_div), heading_style="ATX")

        # Cleanup
        lines = markdown_content.split("\n")
        cleaned_lines = []
        in_nav_table = False
        skip_until_heading = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if skip_until_heading:
                if stripped.startswith("#"):
                    skip_until_heading = False
                else:
                    continue

            if stripped.startswith("#"):
                heading_lower = stripped.lower()
                for unwanted in [
                    "see also",
                    "notes",
                    "references",
                    "further reading",
                    "external links",
                    "bibliography",
                    "citations",
                ]:
                    if unwanted in heading_lower:
                        skip_until_heading = True
                        continue

            if stripped.startswith("|"):
                pipe_count = line.count("|")
                if pipe_count > 3:
                    in_nav_table = True
                    continue

            if in_nav_table:
                if (
                    stripped.startswith("|")
                    or stripped == "---"
                    or (stripped == "" and i < len(lines) - 1)
                ):
                    continue
                else:
                    in_nav_table = False

            if (
                stripped.startswith("* [v]")
                or stripped.startswith("* [t]")
                or stripped.startswith("* [e]")
                or stripped == "* v * t * e"
            ):
                continue

            if (
                stripped.lower().startswith("retrieved from")
                or (stripped.lower().startswith("cite") and len(stripped) < 20)
                or "retrieved from" in stripped.lower()
            ):
                continue

            if stripped == "---" and len(cleaned_lines) > 0:
                prev_line = cleaned_lines[-1].strip() if cleaned_lines else ""
                if prev_line.startswith("|") or (
                    prev_line == "" and len(cleaned_lines) > 1
                ):
                    continue

            if not cleaned_lines and not stripped:
                continue

            cleaned_lines.append(line)

        markdown_content = "\n".join(cleaned_lines)
        markdown_content = remove_urls_and_links(markdown_content)

        lines = markdown_content.split("\n")
        while lines and (
            not lines[-1].strip() or "retrieved from" in lines[-1].strip().lower()
        ):
            lines.pop()

        return "\n".join(lines).strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def remove_urls_and_links(text: str) -> str:
    """Removes all URLs and links from markdown."""

    def replace_image(match):
        alt_text = match.group(1) if match.group(1) else ""
        return f"[Image: {alt_text}]" if alt_text else "[Image]"

    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", replace_image, text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)
    text = re.sub(r"https?://[^\s\)\]\"]+", "", text)
    text = re.sub(r"www\.[^\s\)\]\"]+", "", text)
    text = re.sub(r"\[?/wiki/[^\s\)\]\"]+\]?", "", text)
    text = re.sub(r"[a-zA-Z0-9-]+\.[a-zA-Z]{2,}/[^\s\)\]\"]+", "", text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def fetch_and_save_article(
    wiki_url: str,
    dataset_name: str,
    item_id: str,
    article_idx: int,
    markdown_dir: str,
) -> Tuple[bool, str, str]:
    """Fetch a single Wikipedia article and save it to a file."""
    if "," in wiki_url:
        http_count = wiki_url.count("http")
        if http_count > 1:
            match = re.search(r",\s*https?://", wiki_url)
            if match:
                wiki_url = wiki_url[: match.start()].strip()

    if not wiki_url.startswith("http"):
        return False, wiki_url, ""

    markdown_content = fetch_wikipedia_content(wiki_url)
    if markdown_content:
        filename = f"{dataset_name}_{item_id}_{article_idx}.md"
        markdown_path = os.path.join(markdown_dir, filename)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return True, wiki_url, filename
    return False, wiki_url, ""


def fetch_unique_url(
    wiki_url: str, dataset_name: str, markdown_dir: str
) -> Tuple[bool, str, str]:
    """Fetch a unique Wikipedia URL and save it with a canonical filename."""
    # Normalize URL for filename (use URL path as identifier)
    url_normalized = wiki_url.strip().rstrip("/")
    # Extract article name from URL for filename
    url_parts = url_normalized.split("/")
    article_slug = url_parts[-1] if url_parts else "article"
    # Sanitize filename
    article_slug = re.sub(r"[^\w\-_\.]", "_", article_slug)[:100]  # Limit length

    canonical_filename = f"{dataset_name}_canonical_{article_slug}.md"
    canonical_path = os.path.join(markdown_dir, canonical_filename)

    # Check if already fetched
    if os.path.exists(canonical_path):
        return True, wiki_url, canonical_filename

    if "," in wiki_url:
        http_count = wiki_url.count("http")
        if http_count > 1:
            match = re.search(r",\s*https?://", wiki_url)
            if match:
                wiki_url = wiki_url[: match.start()].strip()

    if not wiki_url.startswith("http"):
        return False, wiki_url, ""

    markdown_content = fetch_wikipedia_content(wiki_url)
    if markdown_content:
        with open(canonical_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return True, wiki_url, canonical_filename
    return False, wiki_url, ""


def process_dataset(
    data: List[Dict], dataset_name: str, markdown_dir: str, max_workers: int = 10
):
    """
    Process dataset: fetch Wikipedia content from wiki_links and save as markdown files.

    Deduplicates URLs across the entire dataset - each unique URL is fetched only once.
    Creates symlinks for items that reference the same URL.
    """
    os.makedirs(markdown_dir, exist_ok=True)

    print(f"\nProcessing {dataset_name}...")

    # Step 1: Collect all URL references and deduplicate
    url_to_references = {}  # url -> list of (item_id, article_idx)
    total_wiki_urls = 0

    for item in data:
        item_id = item.get("Unnamed: 0", "unknown")
        wiki_links = parse_wiki_links_field(item.get("wiki_links", []))
        if wiki_links:
            for article_idx, wiki_url in enumerate(wiki_links):
                # Normalize URL (strip and remove trailing slash)
                normalized_url = wiki_url.strip().rstrip("/")
                if normalized_url not in url_to_references:
                    url_to_references[normalized_url] = []
                url_to_references[normalized_url].append((item_id, article_idx))
            total_wiki_urls += len(wiki_links)

    unique_urls = list(url_to_references.keys())
    print(f"  Total wiki URLs: {total_wiki_urls}")
    print(f"  Unique URLs: {len(unique_urls)}")
    print(f"  Deduplication saved {total_wiki_urls - len(unique_urls)} fetches")

    if not unique_urls:
        print(f"No wiki links found in {dataset_name}")
        return

    # Step 2: Fetch each unique URL once
    print(f"\nFetching {len(unique_urls)} unique URLs...")
    url_to_canonical_file = {}  # url -> canonical_filename
    successful_fetches = 0
    failed_fetches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {}
        for wiki_url in unique_urls:
            future = executor.submit(
                fetch_unique_url,
                wiki_url,
                dataset_name,
                markdown_dir,
            )
            future_to_url[future] = wiki_url

        with tqdm(
            total=len(unique_urls), desc=f"Fetching {dataset_name} articles"
        ) as pbar:
            for future in as_completed(future_to_url):
                wiki_url = future_to_url[future]
                try:
                    success, url, canonical_filename = future.result()
                    if success:
                        url_to_canonical_file[wiki_url] = canonical_filename
                        successful_fetches += 1
                    else:
                        failed_fetches += 1
                except Exception as e:
                    print(f"\nError processing {wiki_url}: {e}")
                    failed_fetches += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix(
                        {"success": successful_fetches, "failed": failed_fetches}
                    )

    # Step 3: Create URL mapping file
    print(f"\nCreating URL mapping file...")
    url_mapping = {}  # (item_id, article_idx) -> canonical_filename

    for wiki_url, references in url_to_references.items():
        if wiki_url not in url_to_canonical_file:
            # URL fetch failed, skip
            continue

        canonical_filename = url_to_canonical_file[wiki_url]
        for item_id, article_idx in references:
            url_mapping[f"{item_id}_{article_idx}"] = canonical_filename

    # Save mapping file
    mapping_file = os.path.join(markdown_dir, f"{dataset_name}_url_mapping.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(url_mapping, f, indent=2, ensure_ascii=False)
    print(f"  Saved mapping: {mapping_file} ({len(url_mapping)} entries)")

    print(f"\n{dataset_name} Summary:")
    print(f"  Total wiki URLs: {total_wiki_urls}")
    print(f"  Unique URLs: {len(unique_urls)}")
    print(f"  Successful fetches: {successful_fetches}")
    print(f"  Failed fetches: {failed_fetches}")
    print(f"  URL mappings: {len(url_mapping)}")
    print(f"  Markdown files saved to: {markdown_dir}")


def main(
    dataset: str = "all",
    input_dir: str = None,
    output_dir: str = None,
    max_workers: int = 20,
):
    """
    Fetch Wikipedia content from wiki_links in test.json and val.json.

    Args:
        dataset: Which dataset to process: "test", "val", or "all" (default: "all")
        input_dir: Directory containing test.json and val.json (default: data/dataset/)
        output_dir: Directory to save markdown files (default: data/markdown/)
        max_workers: Number of parallel threads (default: 20)
    """
    if max_workers > 50:
        print(f"Warning: Using {max_workers} workers may cause rate limiting.")

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if input_dir is None:
        input_dir = os.path.join(project_root, "data", "dataset")

    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "markdown")

    datasets_to_process = []

    if dataset in ["all", "test"]:
        test_path = os.path.join(input_dir, "test.json")
        if os.path.exists(test_path):
            with open(test_path, "r") as f:
                datasets_to_process.append(("test", json.load(f)))
        else:
            print(f"Warning: {test_path} not found")

    if dataset in ["all", "val"]:
        val_path = os.path.join(input_dir, "val.json")
        if os.path.exists(val_path):
            with open(val_path, "r") as f:
                datasets_to_process.append(("val", json.load(f)))
        else:
            print(f"Warning: {val_path} not found")

    if not datasets_to_process:
        print(
            "No datasets found. Run 'uv run tools/dataset/download_dataset.py' first."
        )
        return

    for name, data in datasets_to_process:
        process_dataset(data, name, output_dir, max_workers=max_workers)

    print(f"\nAll markdown files saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
