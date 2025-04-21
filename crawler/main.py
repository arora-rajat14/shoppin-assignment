# crawler/main.py
import asyncio
import argparse
import json
import logging
import re  # Import regex module
import gzip  # Import gzip module
from io import BytesIO  # To handle gzipped content in memory
from typing import List, Dict, Set, Optional, Tuple, Deque  # Added Deque
from urllib.parse import urlparse, urljoin
import httpx
from robotexclusionrulesparser import RobotExclusionRulesParser
from lxml import etree  # Use lxml for efficient XML parsing
from collections import deque  # Use deque for crawl queue

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
USER_AGENT = "ProductURLDiscoveryBot/1.0 (+http://example.com/botinfo)"  # Be polite: Identify your bot

# --- Configurable Patterns ---
# Basic patterns - These will likely need significant refinement per site
# Examples: /product/, /p/, /dp/, /item/, numeric endings?
PRODUCT_URL_PATTERNS = [
    re.compile(r"/(?:products?|p|dp|item)/[\w-]+", re.IGNORECASE),
    re.compile(r"/\d+\.html$", re.IGNORECASE),  # e.g., /12345.html
    re.compile(r"/p-\w+$", re.IGNORECASE),  # e.g., /p-mqaed19764 (TataCliq)
]
# Examples: /category/, /collections/, /dept/, /shop/
CATEGORY_URL_PATTERNS = [
    re.compile(
        r"/(?:category|categories|collections?|dept|shop|c)/[\w-]+", re.IGNORECASE
    ),
    re.compile(r"/browse/", re.IGNORECASE),
]

# --- Utility Functions ---


def normalize_url(url: str, base_domain_netloc: str) -> Optional[str]:
    """Normalizes a URL: adds scheme if missing, lowercases scheme/host, removes fragment."""
    try:
        url = url.strip()
        if not url:
            return None

        # Add scheme if missing, assuming https
        parsed = urlparse(url)
        current_scheme = parsed.scheme.lower()
        current_netloc = parsed.netloc.lower()

        if not current_scheme:
            # Handle //example.com/path URLs
            if url.startswith("//"):
                url = "https://" + url.lstrip("/")
            # Handle relative paths - requires base_domain_netloc to resolve correctly
            elif url.startswith("/"):
                # Use https as default scheme if base domain is missing it somehow
                url = f"https://{base_domain_netloc}{url}"
            else:
                # Attempting to fix incomplete URLs like 'www.example.com/page'
                url = f"https://{url}"

            # Re-parse after potential modification
            parsed = urlparse(url)
            current_scheme = parsed.scheme.lower()  # Update scheme
            current_netloc = parsed.netloc.lower()  # Update netloc

        # Reconstruct URL with lowercase scheme/netloc and no fragment
        # Also removes trailing slash for consistency, except for root domain path '/'
        path = (
            parsed.path.rstrip("/")
            if parsed.path and parsed.path != "/"
            else parsed.path or "/"
        )
        normalized = f"{current_scheme}://{current_netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"  # Keep query params

        return normalized
    except Exception as e:
        logger.warning(f"Failed to normalize URL '{url}': {e}")
        return None


def is_same_domain(url: str, base_domain_netloc: str) -> bool:
    """Checks if the URL belongs to the same base domain."""
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc.lower() == base_domain_netloc.lower()
    except Exception:
        return False


def is_product_url(url: str) -> bool:
    """Checks if the URL matches any known product patterns."""
    # Optimization: Check for common extensions first if needed (e.g., skip .jpg, .css)
    # path = urlparse(url).path
    # if path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.css', '.js', '.pdf')):
    #    return False
    for pattern in PRODUCT_URL_PATTERNS:
        if pattern.search(url):
            return True
    return False


def is_category_url(url: str) -> bool:
    """Checks if the URL matches any known category/listing patterns."""
    for pattern in CATEGORY_URL_PATTERNS:
        if pattern.search(url):
            return True
    return False


def get_robots_url(domain: str) -> str:
    """Constructs the standard /robots.txt URL for a domain."""
    return urljoin(domain, "/robots.txt")


async def fetch_and_parse_robots(
    robots_url: str, client: httpx.AsyncClient
) -> Tuple[Optional[RobotExclusionRulesParser], Optional[str]]:
    """
    Fetches and parses the robots.txt file.
    Returns the parser object and the final URL fetched (after redirects).
    """
    parser = RobotExclusionRulesParser()
    parser.user_agent = (
        USER_AGENT  # Set user agent for the parser internal use (e.g., crawl-delay)
    )
    final_url_fetched = None
    try:
        response = await client.get(robots_url, timeout=10.0)
        final_url_fetched = str(response.url)  # Store the final URL after redirects
        response.raise_for_status()  # Check for 4xx/5xx errors explicitly AFTER getting final URL

        # If we get here, status code is 2xx
        parser.parse(response.text)
        logger.info(
            f"Successfully fetched and parsed robots.txt from {final_url_fetched} (initial: {robots_url})"
        )
        return parser, final_url_fetched

    except httpx.HTTPStatusError as e:
        # Handle non-2xx status codes specifically
        if e.response.status_code == 404:
            logger.warning(
                f"robots.txt not found (404) at {final_url_fetched or robots_url}. Assuming allow all."
            )
            return (
                None,
                final_url_fetched,
            )  # Return None parser, but potentially final URL if redirect happened before 404
        else:
            logger.error(
                f"HTTP error fetching robots.txt from {final_url_fetched or robots_url}. Status: {e.response.status_code}"
            )
            return None, final_url_fetched
    except httpx.RequestError as e:
        logger.error(f"Network error fetching robots.txt from {robots_url}: {e}")
        return None, None  # Network error, no final URL
    except Exception as e:
        logger.error(f"Error processing robots.txt from {robots_url}: {e}")
        return None, None  # Other processing error


def get_sitemap_urls(parser: Optional[RobotExclusionRulesParser]) -> List[str]:
    """Extracts sitemap URLs from a parsed robots.txt parser object."""
    # Check explicitly for sitemaps attribute before accessing
    if parser and hasattr(parser, "sitemaps") and parser.sitemaps:
        sitemap_list = list(parser.sitemaps)
        logger.info(f"Found sitemaps in robots.txt: {sitemap_list}")
        return sitemap_list
    return []


def can_fetch(
    url: str, parser: Optional[RobotExclusionRulesParser], user_agent: str
) -> bool:
    """Checks if the user agent is allowed to fetch the URL based on robots.txt rules."""
    if parser:
        # Always pass both user_agent and url to is_allowed
        return parser.is_allowed(user_agent, url)
    # If no robots.txt was found or parsed, assume we are allowed.
    return True


# --- Sitemap Processing ---

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


async def process_single_sitemap(
    sitemap_url: str, client: httpx.AsyncClient, domain_config: Dict
):
    """Fetches and parses a single sitemap file (XML or XML.GZ)."""
    base_domain_netloc = domain_config["base_domain_netloc"]
    robots_parser = domain_config["robots_parser"]
    user_agent = domain_config["user_agent"]
    discovered_links = domain_config["discovered_links"]
    product_urls = domain_config["product_urls"]
    crawl_queue = domain_config["crawl_queue"]
    processed_sitemaps = domain_config["processed_sitemaps"]

    # Avoid cycles and re-processing
    if sitemap_url in processed_sitemaps:
        return []
    processed_sitemaps.add(sitemap_url)

    logger.info(f"[{base_domain_netloc}] Processing sitemap: {sitemap_url}")

    if not can_fetch(sitemap_url, robots_parser, user_agent):
        logger.warning(
            f"[{base_domain_netloc}] Skipping sitemap disallowed by robots.txt: {sitemap_url}"
        )
        return []

    nested_sitemap_urls = []
    try:
        response = await client.get(
            sitemap_url, timeout=30.0
        )  # Longer timeout for potentially large sitemaps
        response.raise_for_status()  # Raise exception for 4xx/5xx errors

        content = response.content  # Get content as bytes

        # Handle Gzip
        if sitemap_url.endswith(".gz"):
            try:
                content = gzip.decompress(content)
            except gzip.BadGzipFile:
                logger.error(
                    f"[{base_domain_netloc}] Failed to decompress gzipped sitemap: {sitemap_url}"
                )
                return []

        # Use lxml to parse
        try:
            # Use fromstring with bytes content
            tree = etree.fromstring(content)
        except etree.XMLSyntaxError as e:
            logger.error(
                f"[{base_domain_netloc}] Failed to parse XML sitemap {sitemap_url}: {e}"
            )
            return []

        # Check if it's a sitemap index file
        if tree.tag == etree.QName(SITEMAP_NS["sm"], "sitemapindex"):
            logger.info(f"[{base_domain_netloc}] Found sitemap index: {sitemap_url}")
            # Namespace might vary, find 'loc' within 'sitemap' tags robustly
            for sitemap_tag in tree.xpath("//sm:sitemap", namespaces=SITEMAP_NS):
                loc_tag = sitemap_tag.xpath("./sm:loc", namespaces=SITEMAP_NS)
                if loc_tag and loc_tag[0].text:
                    nested_sitemap_urls.append(loc_tag[0].text.strip())
        # Otherwise, assume it's a URL set sitemap
        elif tree.tag == etree.QName(SITEMAP_NS["sm"], "urlset"):
            urls_processed = 0
            # Find 'loc' within 'url' tags robustly
            for url_tag in tree.xpath("//sm:url", namespaces=SITEMAP_NS):
                loc_tag = url_tag.xpath("./sm:loc", namespaces=SITEMAP_NS)
                if loc_tag and loc_tag[0].text:
                    url = loc_tag[0].text.strip()
                    normalized = normalize_url(url, base_domain_netloc)

                    if normalized and is_same_domain(normalized, base_domain_netloc):
                        if normalized not in discovered_links:
                            if can_fetch(normalized, robots_parser, user_agent):
                                discovered_links.add(normalized)
                                urls_processed += 1
                                if is_product_url(normalized):
                                    product_urls.add(normalized)
                                    # logger.debug(f"[{base_domain_netloc}] Found potential product URL via sitemap: {normalized}")
                                elif is_category_url(normalized):
                                    crawl_queue.append(
                                        normalized
                                    )  # Add potential categories to crawl later
                                    # logger.debug(f"[{base_domain_netloc}] Found potential category URL via sitemap: {normalized}")
                            # else:
                            # logger.debug(f"[{base_domain_netloc}] Skipping URL disallowed by robots.txt: {normalized}")
                        # else:
                        # logger.debug(f"[{base_domain_netloc}] Skipping already discovered URL: {normalized}")

            logger.info(
                f"[{base_domain_netloc}] Processed {urls_processed} new URLs from sitemap: {sitemap_url}"
            )

    except httpx.RequestError as e:
        logger.error(
            f"[{base_domain_netloc}] Network error fetching sitemap {sitemap_url}: {e}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[{base_domain_netloc}] HTTP error fetching sitemap {sitemap_url}: {e.response.status_code}"
        )
    except etree.LxmlError as e:  # Catch lxml specific errors
        logger.error(
            f"[{base_domain_netloc}] LXML parsing error for sitemap {sitemap_url}: {e}"
        )
    except Exception as e:
        logger.error(
            f"[{base_domain_netloc}] Unexpected error processing sitemap {sitemap_url}: {e}"
        )

    return nested_sitemap_urls


async def process_sitemaps(
    initial_sitemap_urls: List[str],
    client: httpx.AsyncClient,
    domain_config: Dict,
    max_depth=5,
):
    """Processes a list of sitemaps, handling nested sitemap indexes."""
    sitemap_queue = deque(
        [(url, 0) for url in initial_sitemap_urls]
    )  # Queue of (url, depth)

    while sitemap_queue:
        current_url, current_depth = sitemap_queue.popleft()

        if current_depth > max_depth:
            logger.warning(
                f"[{domain_config['base_domain_netloc']}] Reached max sitemap recursion depth ({max_depth}) at {current_url}, stopping branch."
            )
            continue

        # Process the current sitemap and get any nested sitemap URLs
        # Use asyncio.gather if we want to process sitemaps from the queue concurrently?
        # For simplicity now, process sequentially. Can optimize later.
        nested_urls = await process_single_sitemap(current_url, client, domain_config)

        for nested_url in nested_urls:
            # Check domain and avoid adding already processed ones here if possible
            norm_nested = normalize_url(nested_url, domain_config["base_domain_netloc"])
            if (
                norm_nested
                and is_same_domain(norm_nested, domain_config["base_domain_netloc"])
                and norm_nested not in domain_config["processed_sitemaps"]
            ):
                sitemap_queue.append((norm_nested, current_depth + 1))


# --- Main Processing Logic ---


async def process_domain(
    domain: str, client: httpx.AsyncClient, user_agent: str, concurrency: int
) -> Tuple[str, Set[str]]:
    """
    Handles crawling a single domain: robots.txt, sitemaps, and eventually page crawling.
    """
    logger.info(f"[{domain}] Starting processing...")
    initial_domain_netloc = urlparse(domain).netloc

    # --- Domain Specific Data Structures ---
    domain_config = {
        "base_domain": domain,  # Store initial base domain
        "base_domain_netloc": initial_domain_netloc,  # Store initial netloc
        "user_agent": user_agent,
        "product_urls": set(),
        "discovered_links": set(),
        "crawl_queue": deque(),
        "robots_parser": None,
        "processed_sitemaps": set(),
    }

    # 1. Fetch and Parse robots.txt
    robots_url = get_robots_url(domain)
    robots_parser, final_robots_url = await fetch_and_parse_robots(robots_url, client)
    domain_config["robots_parser"] = robots_parser

    # Update base_domain_netloc if redirect occurred during robots.txt fetch
    if final_robots_url:
        try:
            final_netloc = urlparse(final_robots_url).netloc
            if final_netloc and final_netloc != initial_domain_netloc:
                logger.info(
                    f"[{domain}] Domain redirected during robots.txt fetch. Updating netloc from '{initial_domain_netloc}' to '{final_netloc}' for checks."
                )
                domain_config["base_domain_netloc"] = (
                    final_netloc  # Use the final netloc for subsequent checks
                )
        except Exception as e:
            logger.warning(
                f"[{domain}] Could not parse final robots URL '{final_robots_url}' to update netloc: {e}"
            )
            # Keep using initial_domain_netloc if parsing fails

    # Use the potentially updated netloc for checks
    current_netloc = domain_config["base_domain_netloc"]

    # Check if base domain itself is disallowed
    # Use the original domain URL for this check as that's the entry point
    if not can_fetch(domain, robots_parser, user_agent):
        logger.warning(
            f"[{domain}] Crawling disallowed by robots.txt for entry URL {domain}. Skipping."
        )
        return domain, domain_config["product_urls"]

    # 2. Extract Sitemap URLs from robots.txt
    sitemap_urls_from_robots = get_sitemap_urls(robots_parser)
    # Ensure sitemaps belong to the current domain (after potential redirect) before processing
    valid_sitemap_urls = [
        s_url
        for s_url in sitemap_urls_from_robots
        if normalize_url(s_url, current_netloc)
        and is_same_domain(normalize_url(s_url, current_netloc), current_netloc)
    ]
    if len(valid_sitemap_urls) < len(sitemap_urls_from_robots):
        logger.warning(
            f"[{current_netloc}] Filtered out {len(sitemap_urls_from_robots) - len(valid_sitemap_urls)} sitemap(s) not matching target domain."
        )

    logger.info(
        f"[{current_netloc}] Found {len(valid_sitemap_urls)} valid sitemap(s) in robots.txt: {valid_sitemap_urls}"
    )

    # 3. Process Sitemaps
    if valid_sitemap_urls:
        await process_sitemaps(valid_sitemap_urls, client, domain_config)
    else:
        logger.info(f"[{current_netloc}] No valid sitemaps found in robots.txt.")
        common_sitemap = urljoin(
            f"https://{current_netloc}", "/sitemap.xml"
        )  # Use current_netloc scheme
        logger.info(
            f"[{current_netloc}] Attempting common sitemap location: {common_sitemap}"
        )
        if can_fetch(common_sitemap, robots_parser, user_agent):
            await process_sitemaps([common_sitemap], client, domain_config)

    # TODO - Step 4: Implement crawling logic

    logger.info(
        f"[{current_netloc}] Finished sitemap processing. Found {len(domain_config['product_urls'])} potential products via sitemaps."
    )
    logger.info(
        f"[{current_netloc}] Found {len(domain_config['crawl_queue'])} potential category/crawl URLs via sitemaps."
    )

    return (
        domain,
        domain_config["product_urls"],
    )  # Return original domain key, but processed data


async def main(domains: List[str], output_file: str, concurrency: int):
    """
    Main asynchronous function to coordinate the crawling process.
    """
    logger.info(f"Starting crawler for domains: {domains}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Concurrency limit (per domain): {concurrency}")

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    timeouts = httpx.Timeout(10.0, read=30.0, write=10.0, pool=5.0)
    total_concurrency = concurrency * len(domains)
    limits = httpx.Limits(
        max_connections=total_concurrency + 20,
        max_keepalive_connections=concurrency + 5,
    )

    async with httpx.AsyncClient(
        http2=True,
        follow_redirects=True,
        headers=headers,
        timeout=timeouts,
        limits=limits,
    ) as client:
        tasks = []
        initial_domains = (
            {}
        )  # Store mapping from potentially modified base_domain back to original input

        for i, domain_url_input in enumerate(domains):
            domain_url = domain_url_input
            if not domain_url.startswith(("http://", "https://")):
                domain_url = f"https://{domain_url}"
            try:
                parsed_uri = urlparse(domain_url)
                base_domain = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                initial_domains[base_domain] = (
                    domain_url_input  # Map clean base domain to original input
                )

                # Add a small delay between starting tasks for different domains
                if i > 0:
                    await asyncio.sleep(0.1)  # 100ms delay

                tasks.append(
                    asyncio.create_task(
                        process_domain(base_domain, client, USER_AGENT, concurrency)
                    )
                )
            except Exception as e:
                logger.error(
                    f"Failed to create task for domain '{domain_url_input}': {e}"
                )

        domain_results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        final_results = {}
        for result in domain_results_list:
            original_domain_key = "Unknown"  # Fallback key
            try:
                if isinstance(result, Exception):
                    # Attempt to find original domain based on exception context if possible (difficult)
                    logger.error(f"Task for a domain failed with exception: {result}")
                    # Add more specific error logging if possible, e.g., by wrapping process_domain
                elif result:
                    domain_key, product_urls = (
                        result  # domain_key is the initial base_domain passed to process_domain
                    )
                    original_domain_key = initial_domains.get(
                        domain_key, domain_key
                    )  # Map back to original input domain if possible
                    if product_urls:
                        final_results[original_domain_key] = sorted(list(product_urls))
                else:
                    logger.warning(
                        f"No result or empty result returned for a domain task."
                    )
            except Exception as e:
                logger.error(
                    f"Error processing result for domain {original_domain_key}: {e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce Product URL Crawler")
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        required=True,
        help="List of domain names to crawl (e.g., virgio.com tatacliq.com)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="product_urls.json",
        help="Output JSON file path (default: product_urls.json)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests *per domain* (default: 5)",
    )  # Changed default and help text

    args = parser.parse_args()

    try:
        asyncio.run(main(args.domains, args.output, args.concurrency))
    except KeyboardInterrupt:
        logger.info("Crawler interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Crawler finished.")
