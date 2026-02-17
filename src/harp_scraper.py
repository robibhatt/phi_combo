"""HARP 2025 dataset scraper for AMC/AIME problems from AoPS wiki."""

import html
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


class HarpScraperError(Exception):
    """Raised when scraping fails."""

    pass


# AoPS Wiki URL template
AOPS_URL_TEMPLATE = (
    "https://artofproblemsolving.com/wiki/index.php/{year}_{contest}_Problems/Problem_{number}"
)

# Firefox user-agent headers for requests
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


def build_url(year: int, contest: str, number: int) -> str:
    """Build AoPS wiki URL for a problem.

    Args:
        year: Competition year.
        contest: Contest name (e.g., "AMC_10A", "AIME_I").
        number: Problem number.

    Returns:
        Full URL to the problem page.
    """
    return AOPS_URL_TEMPLATE.format(year=year, contest=contest, number=number)


def get_2025_contests() -> list[dict[str, Any]]:
    """Get contest definitions for 2025.

    Returns:
        List of contest dictionaries with name and problems count.
    """
    return [
        {"name": "AMC_8", "problems": 25},
        {"name": "AMC_10A", "problems": 25},
        {"name": "AMC_10B", "problems": 25},
        {"name": "AMC_12A", "problems": 25},
        {"name": "AMC_12B", "problems": 25},
        {"name": "AIME_I", "problems": 15},
        {"name": "AIME_II", "problems": 15},
    ]


def has_choices(contest: str) -> bool:
    """Check if contest has multiple choice answers.

    Args:
        contest: Contest name.

    Returns:
        True if contest has A-E choices, False otherwise.
    """
    return contest.startswith("AMC")


def replace_img_alt_text(s: str) -> str:
    """Extract img block alt text.

    For latex images, this contains the source latex code.
    For actual images, this is the file name or a caption.

    Args:
        s: HTML string with img tags.

    Returns:
        String with img tags replaced by their alt text (if LaTeX).
    """

    def _replace_one(alt_text: str) -> str:
        if bool(re.fullmatch(r"^.*\.(png|jpeg|jpg|gif)$", alt_text, flags=re.IGNORECASE)):
            return ""

        if (
            (alt_text.startswith("$") and alt_text.endswith("$"))
            or (alt_text.startswith("\\[") and alt_text.endswith("\\]"))
            or (alt_text.startswith("\\(") and alt_text.endswith("\\)"))
            or (alt_text.startswith("\\begin{"))
            or (alt_text.startswith("[asy]") and alt_text.endswith("[/asy]"))
        ):
            if "$" in alt_text[1:-1] and not alt_text.startswith("[asy]"):
                alt_text = alt_text[0] + re.sub(r"\$", r"\\textdollar", alt_text[1:-1]) + alt_text[-1]
            return alt_text

        return ""

    return re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', lambda x: _replace_one(x.group(1)), s)


def standardize_boxed_command(text: str) -> str:
    """Standardize boxed commands to \\boxed{}.

    Args:
        text: Text containing various boxed command formats.

    Returns:
        Text with standardized \\boxed{} commands.
    """
    text = re.sub(r"\\fbox", r"\\boxed", text)
    text = re.sub(r"\\framebox(\[.*?\])?", r"\\boxed", text)
    text = re.sub(r"\\boxed +{", r"\\boxed{", text)
    return text


def replace_unicode_chars(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents.

    Args:
        text: Text containing Unicode characters.

    Returns:
        Text with Unicode characters replaced.
    """
    text = re.sub("\u00a0", " ", text)
    text = re.sub("\u200b", "", text)
    text = re.sub("\u2018", "'", text)
    text = re.sub("\u2019", "'", text)
    text = re.sub("\u201c", "'", text)
    text = re.sub("\u201d", "'", text)
    text = re.sub("\uff0c", ", ", text)
    text = re.sub("\u2013", "-", text)
    text = re.sub("\u2014", "--", text)
    text = re.sub("\u301c", "~", text)
    text = re.sub("\u00a9", "(C)", text)
    return text


def find_closing_brace(text: str) -> int:
    """Find the index of the closing brace matching an opening brace at position 0.

    Assumes we start right after an opening brace.

    Args:
        text: Text to search in.

    Returns:
        Index of closing brace, or -1 if not found.
    """
    cnt = 1
    for i, c in enumerate(text):
        if c == "{":
            cnt += 1
        if c == "}":
            cnt -= 1
            if cnt == 0:
                return i
    return -1


def parse_html_to_text(raw_html: str) -> str:
    """Parse HTML and extract problem/solution text.

    Args:
        raw_html: Raw HTML content from AoPS wiki.

    Returns:
        Parsed text with problem and solutions.
    """
    retval = []

    # Remove the table of contents element
    if bool(toc_match := re.search(r'<div [^>]*?"toc"[^>]?>', raw_html)):
        start_idx = toc_match.span()[1]
        cnt = 1
        for i, c in enumerate(raw_html[start_idx:]):
            if c == "<":
                if raw_html[start_idx + i : start_idx + i + 4] == "<div":
                    cnt += 1
                elif raw_html[start_idx + i : start_idx + i + 6] == "</div>":
                    cnt -= 1
                if cnt == 0:
                    break
        raw_html = raw_html[: start_idx].strip() + raw_html[start_idx + i + 6 :].strip()

    raw_html = re.sub(r"\s*<hr />\s*", "", raw_html)
    raw_html = re.sub(r'<p class="mw-empty-elt">\s*</p>', "", raw_html)
    raw_html = re.sub(r"center>", "p>", raw_html)
    raw_html = re.sub(
        r'(<div class="(center|(float(right|left|none)))">)+(.+?)(</div>)+',
        r"<p>\5</p>",
        raw_html,
    )
    raw_html = re.sub(r'(<div style="text-align:center;?">)+(.+?)(</div>)+', r"<p>\2</p>", raw_html)

    # Handle lists
    def convert_ordered_list(s: str) -> str:
        converted = []
        for i, (tag, elem) in enumerate(re.findall(r"<(li|p)>(.*?)</\1>", s, flags=re.DOTALL)):
            elem = elem.strip()
            elem = re.sub(r"::marker\s*", "", elem)
            elem = re.sub(r"</?p>", "", elem)
            if tag == "li":
                converted.append(f"{i+1}. {elem}")
            else:
                converted.append(elem)
        return "<p>" + "\n".join(converted) + "</p>"

    raw_html = re.sub(
        r"<ol([^>]*?)>(.+?)</ol>",
        lambda x: convert_ordered_list(x.group(2)),
        raw_html,
        flags=re.DOTALL,
    )
    raw_html = re.sub(r"<(/)?[du]l([^>]*?)>", r"<\1p>", raw_html)
    raw_html = re.sub(r"<li>(.+?)</li>", r"* \1\n", raw_html)
    raw_html = re.sub(r"<dd>(.+?)</dd>", r"\1\n", raw_html)
    raw_html = re.sub(r"<dt>(.+?)</dt>", r"\1\n", raw_html)
    raw_html = re.sub(r"::marker\s*", "", raw_html)

    # Collapse nested <p> blocks
    raw_html = re.sub(r"<p>\s*<p>", r"<p>", raw_html)
    raw_html = re.sub(r"</p>\s*</p>", r"</p>", raw_html)
    raw_html = re.sub(r"</p><p>", "", raw_html)
    raw_html = re.sub(r"</p>\n+<p>", "\n", raw_html)
    raw_html = re.sub(r"</p>\s+<p>", " ", raw_html)

    raw_html = re.sub(r"<br />", "\n", raw_html)
    raw_html = re.sub(r'<a (.*?)href=".+?>(.+?)</a>', r"\2", raw_html)

    for a, b in re.findall(
        r">([^<]*Problem ?#?\d*|Solution[^<]*)<[^\n]*\n<p>(.*?)</p>", raw_html, re.DOTALL
    ):
        if a.startswith("Problem"):
            a = "Problem"
        retval.append("# " + a)
        parsed = replace_img_alt_text(b)
        parsed = re.sub(r"^~.*", "", parsed, flags=re.MULTILINE)
        parsed = re.sub(r"\s[~|-][a-z|A-Z][-_\w]*?$", "", parsed, flags=re.MULTILINE)
        parsed = re.sub(r"<i>Alternate solutions.*?</i>", "", parsed)
        parsed = standardize_boxed_command(parsed)
        parsed = html.unescape(parsed)
        parsed = replace_unicode_chars(parsed)
        parsed = re.sub(r"\\\[\\\]", "", parsed)
        parsed = re.sub(r"\[mathjax display=true\](.*?)\[/mathjax\]", r"\[\1\]", parsed)
        parsed = re.sub(r"\[mathjax\](.*?)\[/mathjax\]", r"$\1$", parsed)
        parsed = re.sub(r"\n+", "\n", parsed)
        retval.append(parsed.strip())

    return "\n\n".join(retval)


def clean_choice_format(text: str) -> str:
    """Clean up various choice format inconsistencies.

    Args:
        text: Problem text with choices.

    Returns:
        Text with standardized choice format.
    """
    for i in range(5):
        c = chr(65 + i)
        if f"(\\mathrm{{{c}}})" in text:
            text = re.sub(r"\(\\mathrm{(\w)}\)", r"\\mathrm{(\1)}", text)
        if f"(\\mathrm {{{c}}})" in text:
            text = re.sub(r"\(\\mathrm {(\w)}\)", r"\\mathrm{(\1)}", text)
        if f"\\textbf {{({c})" in text:
            text = re.sub(r"\\textbf {\((\w)\)", r"\\textbf{(\1)", text)
        if f"\\text {{({c})" in text:
            text = re.sub(r"\\text {\((\w)\)", r"\\text{(\1)", text)
    return text


def extract_choices(text: str) -> dict[str, Any] | None:
    """Extract multiple choice options from problem text.

    Args:
        text: Problem text containing choices.

    Returns:
        Dictionary with choices, start index, and end index, or None if not found.
    """
    format_str = None
    choices_start = -1
    for poss in ["\\textbf{", "\\text{", "\\mathrm{"]:
        poss_start = text.find("$" + poss + "(A)")
        if poss_start >= 0:
            format_str = poss
            choices_start = poss_start
            break

    if format_str is None:
        return None

    choices_end = text.find("\n", text.find(format_str + "(E)"))
    if choices_end == -1:
        choices_end = len(text) - 1

    retval = dict()
    for c in range(5):
        look_for = format_str + "({choice})".format(choice=chr(65 + c))
        index = text.find(look_for)
        if index == -1:
            return None

        index += len(look_for)
        remaining_text = text[index:]

        brace_idx = find_closing_brace(remaining_text)
        if brace_idx == -1:
            return None

        raw_before_brace = remaining_text[:brace_idx].strip()
        if c < 4:
            next_look = format_str + "({choice})".format(choice=chr(65 + c + 1))
            end = remaining_text.find(next_look)
            raw_after_brace = remaining_text[brace_idx + 1 : end]
            alt_end = remaining_text.find("\n", brace_idx + 3)
            if alt_end != -1 and alt_end < end:
                raw_after_brace = remaining_text[brace_idx + 1 : alt_end]
        else:
            end = remaining_text.find("\n", brace_idx + 3)
            if end == -1:
                raw_after_brace = remaining_text[brace_idx + 1 :]
            else:
                raw_after_brace = remaining_text[brace_idx + 1 : end]

        raw_choice = raw_before_brace + " " + raw_after_brace.strip()
        choice = re.sub(r"[\s\$]*$", "", raw_choice).strip()

        if choice.startswith("$"):
            choice = choice[1:].rstrip("$").strip()
            retval[chr(65 + c)] = choice
            continue

        choice = "${}$".format(choice)
        if "$" in choice[1:-1]:
            choice = re.sub(r"\$\$", "", choice).strip()
        retval[chr(65 + c)] = choice

    return {"choices": retval, "choices_start_ind": choices_start, "choices_end_ind": choices_end + 1}


def extract_last_boxed_from_text(text: str) -> str | None:
    """Extract content from the last \\boxed{} command in text.

    Args:
        text: Text containing \\boxed{} commands.

    Returns:
        Content inside the last \\boxed{}, or None if not found.
    """
    prefix = "\\boxed{"

    ind = text.rfind(prefix)
    if ind == -1:
        return None

    close_idx = find_closing_brace(text[ind + len(prefix) :])
    if close_idx == -1:
        return None

    return text[ind + len(prefix) : ind + len(prefix) + close_idx]


def map_difficulty(year: str, contest: str, number: int) -> int:
    """Map problem to difficulty level 1-6.

    Based on AoPS wiki competition ratings.

    Args:
        year: Competition year (as string).
        contest: Contest name.
        number: Problem number.

    Returns:
        Difficulty level from 1 (easiest) to 6 (hardest).
    """
    if contest.startswith("AMC_8"):
        if number <= 20:
            return 1
        else:
            return 2
    elif contest.startswith("AMC_10"):
        if number <= 5:
            return 1
        elif number <= 20:
            return 2
        else:
            return 3
    elif contest.startswith("AMC_12"):
        if number <= 10:
            return 2
        elif number <= 20:
            return 3
        else:
            return 4
    elif contest.startswith("AIME"):
        if number <= 5:
            return 3
        elif number <= 9:
            return 4
        elif number <= 12:
            return 5
        else:
            return 6
    return 1


def map_subject(contest: str) -> str:
    """Infer subject from contest type.

    Args:
        contest: Contest name.

    Returns:
        Subject classification.
    """
    if contest.startswith("AMC_8"):
        return "prealgebra"
    return "algebra"


def fetch_problem_page(
    url: str,
    session: requests.Session | None = None,
    retry_count: int = 10,
    retry_backoff: float = 60,
) -> requests.Response | None:
    """Fetch a problem page with retry logic.

    Args:
        url: URL to fetch.
        session: Optional requests session (not used, kept for API compatibility).
        retry_count: Number of retries on failure.
        retry_backoff: Seconds to wait between retries.

    Returns:
        Response object, or None if page not found (404).
    """
    for attempt in range(retry_count):
        try:
            print(f"    [fetch] Attempt {attempt + 1}/{retry_count}: GET {url}", flush=True)
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            print(f"    [fetch] Response status: {response.status_code}", flush=True)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"    [fetch] Request failed: {e}", flush=True)
            if attempt < retry_count - 1:
                print(f"    [fetch] Waiting {retry_backoff}s before retry...", flush=True)
                time.sleep(retry_backoff)
                continue
            raise HarpScraperError(f"Failed to fetch {url} after {retry_count} attempts")
    return None


def extract_problem_from_html(raw_html: str, contest: str) -> dict[str, Any] | None:
    """Extract problem text and choices from HTML.

    Args:
        raw_html: Raw HTML content.
        contest: Contest name.

    Returns:
        Dictionary with problem text and choices, or None if extraction failed.
    """
    parsed = parse_html_to_text(raw_html)
    pieces = re.split(r"\n{2,}#", parsed)
    pieces = [("#" + p if i > 0 else p) for i, p in enumerate(pieces)]

    if not pieces or not pieces[0].startswith("# Problem"):
        return None

    problem_text = "\n".join(pieces[0].split("\n")[1:]).strip()
    problem_text = re.sub(r"^\$(\*)\$\s*", "", problem_text)

    result: dict[str, Any] = {
        "problem": problem_text,
        "choices": None,
        "answer_choice": None,
    }

    if has_choices(contest):
        problem_text = clean_choice_format(problem_text)
        extracted_choices = extract_choices(problem_text)
        if extracted_choices is not None:
            result["choices"] = extracted_choices["choices"]
            # Remove choices from problem text
            problem_text = (
                problem_text[: extracted_choices["choices_start_ind"]]
                + problem_text[extracted_choices["choices_end_ind"] :]
            )
        result["problem"] = re.sub(r"\s+", " ", problem_text).strip()
    else:
        result["problem"] = re.sub(r"\s+", " ", problem_text).strip()

    return result


def scrape_single_problem(
    year: int,
    contest: str,
    number: int,
    request_delay: float = 2.0,
    retry_count: int = 10,
    retry_backoff: float = 60,
) -> dict[str, Any] | None:
    """Scrape a single problem from AoPS wiki.

    Args:
        year: Competition year.
        contest: Contest name.
        number: Problem number.
        request_delay: Delay before making request.
        retry_count: Number of retries.
        retry_backoff: Backoff time between retries.

    Returns:
        Problem dictionary in HARP format, or None if not found.
    """
    print(f"  [delay] Sleeping {request_delay}s before request...", flush=True)
    time.sleep(request_delay)
    print(f"  [delay] Done sleeping", flush=True)

    url = build_url(year, contest, number)
    response = fetch_problem_page(url, retry_count=retry_count, retry_backoff=retry_backoff)

    if response is None:
        print(f"  [parse] No response (404 or failed)", flush=True)
        return None

    print(f"  [parse] Parsing HTML ({len(response.text)} bytes)...", flush=True)
    extracted = extract_problem_from_html(response.text, contest)
    if extracted is None:
        print(f"  [parse] Failed to extract problem from HTML", flush=True)
        return None
    print(f"  [parse] Successfully extracted problem", flush=True)

    # Build HARP format output
    result = {
        "year": str(year),
        "contest": contest,
        "number": number,
        "level": map_difficulty(str(year), contest, number),
        "subject": map_subject(contest),
        "multiple_choice_only": False,
        "problem": extracted["problem"],
        "answer": "",
        "choices": extracted["choices"],
        "answer_choice": extracted.get("answer_choice"),
        "solution_1": "",
        "num_solutions": 0,
    }

    return result


def scrape_all_2025_problems(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scrape all 2025 AMC/AIME problems.

    Args:
        config: Configuration dictionary with contests and options.

    Returns:
        Tuple of (list of problem dictionaries in HARP format, statistics dict).
    """
    results = []
    contests = config.get("contests", get_2025_contests())
    request_delay = config.get("request_delay_seconds", 2)
    retry_count = config.get("retry_count", 10)
    retry_backoff = config.get("retry_backoff_seconds", 60)

    # Calculate totals
    total_problems = sum(c["problems"] for c in contests)
    print(
        f"Will scrape {total_problems} problems across {len(contests)} contests",
        flush=True,
    )

    # Statistics counters
    stats = {"found": 0, "skipped": 0, "not_found": 0, "errors": 0}

    # Check for existing output to resume
    output_dir = Path(config.get("output_dir", "."))
    output_file = output_dir / config.get("output_filename", "HARP_2025.jsonl")
    existing_problems = set()

    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    prob = json.loads(line)
                    key = (prob["contest"], prob["number"])
                    existing_problems.add(key)
                    results.append(prob)
        print(f"Resuming: found {len(existing_problems)} existing problems", flush=True)

    current = 0
    for contest in contests:
        contest_name = contest["name"]
        num_problems = contest["problems"]

        print(f"\n=== {contest_name} ({num_problems} problems) ===", flush=True)

        for num in range(1, num_problems + 1):
            current += 1
            if (contest_name, num) in existing_problems:
                print(
                    f"[{current}/{total_problems}] Skipping {contest_name} #{num} "
                    "(already scraped)",
                    flush=True,
                )
                stats["skipped"] += 1
                continue

            print(
                f"[{current}/{total_problems}] Scraping {contest_name} #{num}...",
                flush=True,
            )
            try:
                problem = scrape_single_problem(
                    2025,
                    contest_name,
                    num,
                    request_delay=request_delay,
                    retry_count=retry_count,
                    retry_backoff=retry_backoff,
                )
                if problem:
                    results.append(problem)
                    stats["found"] += 1
                    print(f"  Found: {problem['problem'][:50]}...", flush=True)
                else:
                    stats["not_found"] += 1
                    print(f"  Not found (404 or parse error)", flush=True)
            except HarpScraperError as e:
                stats["errors"] += 1
                print(f"  Error: {e}", flush=True)

    # Print summary
    print("\n=== Scraping Summary ===", flush=True)
    print(f"  Found:     {stats['found']}", flush=True)
    print(f"  Skipped:   {stats['skipped']}", flush=True)
    print(f"  Not found: {stats['not_found']}", flush=True)
    print(f"  Errors:    {stats['errors']}", flush=True)

    return results, stats


def save_to_jsonl(problems: list[dict[str, Any]], output_path: Path) -> None:
    """Save problems to JSONL file.

    Args:
        problems: List of problem dictionaries.
        output_path: Path to output file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem) + "\n")
