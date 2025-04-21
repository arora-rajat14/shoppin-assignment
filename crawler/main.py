# crawler/main.py
import asyncio
import argparse
import json
import logging
import re
import gzip
from io import BytesIO
from typing import List, Dict, Set, Optional, Tuple, Deque
from urllib.parse import urlparse, urljoin
import httpx
from robotexclusionrulesparser import RobotExclusionRulesParser
from lxml import etree
from collections import deque
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
USER_AGENT = "ProductURLDiscoveryBot/1.0 (+http://example.com/botinfo)"

PRODUCT_URL_PATTERNS = [
    re.compile(
        r"/(?:products?|p|dp|item)/[\w-]+", re.IGNORECASE
    ),  # Common path segments
    re.compile(
        r"/\d{5,}", re.IGNORECASE
    ),  # Contains 5+ digits (potential product ID) in path
    re.compile(r"/\d+\.html$", re.IGNORECASE),  # e.g., /12345.html
    re.compile(r"/p-\w+$", re.IGNORECASE),  # e.g., /p-mqaed19764 (TataCliq style)
]
# Patterns to identify pages likely containing lists of products (to crawl)
CATEGORY_URL_PATTERNS = [
    re.compile(
        r"/(?:category|categories|collections?|dept|shop|c|browse|all)/", re.IGNORECASE
    ),
    re.compile(r"page=\d+", re.IGNORECASE),  # Pagination links
    re.compile(
        r"\?.*(?:cat|dept|collection)=", re.IGNORECASE
    ),  # Query params indicating category
]

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def normalize_url(url: str, base_domain_netloc: str) -> Optional[str]:
    """
    Normalizes a URL: adds scheme if missing, lowercases scheme/host, removes fragment (#...).
    Attempts basic fix for scheme-relative URLs (//) and root-relative paths (/).
    """
    try:
        url = url.strip()
        if not url:
            return None

        # Add scheme if missing, assuming https
        parsed = urlparse(url)
        current_scheme = parsed.scheme.lower()
        current_netloc = parsed.netloc.lower()

        if not current_scheme:
            if url.startswith("//"):
                url = "https://" + url.lstrip("/")
            elif url.startswith("/"):
                # Ensure base_domain_netloc is valid before joining
                if not base_domain_netloc:
                    logger.warning(
                        f"Cannot resolve relative path '{url}' without valid base domain netloc."
                    )
                    return None
                url = f"https://{base_domain_netloc}{url}"
            else:
                # Attempting to fix incomplete URLs like 'www.example.com/page'
                # This is heuristic and might fail for complex relative URLs without a proper base.
                url = f"https://{url}"

            # Re-parse after potential modification
            parsed = urlparse(url)
            current_scheme = parsed.scheme.lower()
            current_netloc = parsed.netloc.lower()
            if not current_scheme or not current_netloc:  # Check if fix worked
                logger.warning(f"Failed to derive scheme/netloc for URL: {url}")
                return None

        # Reconstruct URL with lowercase scheme/netloc and no fragment
        # Also removes trailing slash for consistency, except for root domain path '/'
        path = (
            parsed.path.rstrip("/")
            if parsed.path and parsed.path != "/"
            else parsed.path or "/"
        )
        normalized = f"{current_scheme}://{current_netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"

        return normalized
    except Exception as e:
        logger.warning(f"Failed to normalize URL '{url}': {e}")
        return None


def is_same_domain(url: str, base_domain_netloc: str) -> bool:
    """Checks if the URL's network location matches the base domain's network location."""
    try:
        parsed_url = urlparse(url)
        # Ensure comparison is case-insensitive
        return parsed_url.netloc.lower() == base_domain_netloc.lower()
    except Exception:
        return False


def is_product_url(url: str) -> bool:
    """Checks if the URL matches any known product patterns."""
    for pattern in PRODUCT_URL_PATTERNS:
        if pattern.search(url):
            return True
    return False


def is_category_url(url: str) -> bool:
    """Checks if the URL matches any known category/listing patterns."""
    if is_product_url(url):
        return False
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
    parser.user_agent = USER_AGENT
    final_url_fetched = None
    try:
        response = await client.get(robots_url, timeout=10.0)
        final_url_fetched = str(response.url)
        response.raise_for_status()

        parser.parse(response.text)
        logger.info(
            f"Successfully fetched/parsed robots.txt from {final_url_fetched} (initial: {robots_url})"
        )
        return parser, final_url_fetched

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(
                f"robots.txt not found (404) at {final_url_fetched or robots_url}. Assuming allow all."
            )
        else:
            logger.error(
                f"HTTP error {e.response.status_code} fetching robots.txt from {final_url_fetched or robots_url}."
            )
        return None, final_url_fetched  # Return None parser, but potentially final URL
    except httpx.RequestError as e:
        logger.error(f"Network error fetching robots.txt from {robots_url}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error processing robots.txt from {robots_url}: {e}")
        return None, None


def get_sitemap_urls(parser: Optional[RobotExclusionRulesParser]) -> List[str]:
    """Extracts sitemap URLs from a parsed robots.txt parser object."""
    if parser and hasattr(parser, "sitemaps") and parser.sitemaps:
        sitemap_list = list(parser.sitemaps)
        logger.debug(f"Found sitemaps in robots.txt: {sitemap_list}")
        return sitemap_list
    return []


def can_fetch(
    url: str, parser: Optional[RobotExclusionRulesParser], user_agent: str
) -> bool:
    """Checks if the user agent is allowed to fetch the URL based on robots.txt rules."""
    if parser:
        try:
            # Pass both user_agent and url to is_allowed
            return parser.is_allowed(user_agent, url)
        except Exception as e:
            logger.error(f"Error checking robots allowance for {url}: {e}")
            return False
    return True


async def process_single_sitemap(
    sitemap_url: str, client: httpx.AsyncClient, domain_config: Dict
):
    """
    Fetches and parses a single sitemap file (XML or XML.GZ).
    Updates domain_config with discovered URLs.
    Returns a list of URLs if it's a sitemap index, otherwise empty list.
    """
    base_domain_netloc = domain_config["base_domain_netloc"]
    robots_parser = domain_config["robots_parser"]
    user_agent = domain_config["user_agent"]
    discovered_links = domain_config["discovered_links"]
    product_urls = domain_config["product_urls"]
    crawl_queue = domain_config["crawl_queue"]
    processed_sitemaps = domain_config["processed_sitemaps"]

    if sitemap_url in processed_sitemaps:
        logger.debug(
            f"[{base_domain_netloc}] Skipping already processed sitemap: {sitemap_url}"
        )
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
        response = await client.get(sitemap_url, timeout=60.0)
        final_sitemap_url = str(response.url)
        response.raise_for_status()

        content = response.content

        if response.headers.get(
            "content-encoding"
        ) == "gzip" or final_sitemap_url.endswith(".gz"):
            try:
                content = gzip.decompress(content)
            except gzip.BadGzipFile:
                logger.error(
                    f"[{base_domain_netloc}] Failed to decompress gzipped sitemap: {final_sitemap_url}"
                )
                return []

        try:
            content = re.sub(b"^<\\?xml.*?\\?>", b"", content).strip()
            if not content:
                logger.warning(
                    f"[{base_domain_netloc}] Sitemap content empty after stripping XML decl: {final_sitemap_url}"
                )
                return []
            tree = etree.fromstring(content)
        except etree.XMLSyntaxError as e:
            logger.error(
                f"[{base_domain_netloc}] Failed to parse XML sitemap {final_sitemap_url}: {e}"
            )
            return []

        root_tag = etree.QName(tree.tag)
        root_ns = root_tag.namespace

        ns_map = {"sm": root_ns} if root_ns else SITEMAP_NS

        if root_tag.localname == "sitemapindex":
            logger.info(
                f"[{base_domain_netloc}] Found sitemap index: {final_sitemap_url}"
            )
            for sitemap_tag in tree.xpath("//sm:sitemap", namespaces=ns_map):
                loc_tag = sitemap_tag.xpath("./sm:loc", namespaces=ns_map)
                if loc_tag and loc_tag[0].text:
                    nested_sitemap_urls.append(loc_tag[0].text.strip())
        elif root_tag.localname == "urlset":
            urls_processed_count = 0
            for url_tag in tree.xpath("//sm:url", namespaces=ns_map):
                loc_tag = url_tag.xpath("./sm:loc", namespaces=ns_map)
                if loc_tag and loc_tag[0].text:
                    url = loc_tag[0].text.strip()
                    normalized = normalize_url(url, base_domain_netloc)

                    if normalized and is_same_domain(normalized, base_domain_netloc):
                        if normalized not in discovered_links:
                            if can_fetch(normalized, robots_parser, user_agent):
                                discovered_links.add(normalized)
                                urls_processed_count += 1
                                if is_product_url(normalized):
                                    product_urls.add(normalized)
                                elif is_category_url(normalized):
                                    crawl_queue.append(normalized)
            logger.info(
                f"[{base_domain_netloc}] Processed {urls_processed_count} new URLs from sitemap: {final_sitemap_url}"
            )
        else:
            logger.warning(
                f"[{base_domain_netloc}] Unknown root tag '{tree.tag}' in sitemap: {final_sitemap_url}"
            )

    except httpx.RequestError as e:
        logger.error(
            f"[{base_domain_netloc}] Network error fetching sitemap {sitemap_url}: {e}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[{base_domain_netloc}] HTTP error {e.response.status_code} fetching sitemap {sitemap_url}"
        )
    except etree.LxmlError as e:
        logger.error(
            f"[{base_domain_netloc}] LXML parsing error for sitemap {sitemap_url}: {e}"
        )
    except Exception as e:
        logger.error(
            f"[{base_domain_netloc}] Unexpected error processing sitemap {sitemap_url}: {e}",
            exc_info=True,
        )

    return nested_sitemap_urls


async def process_sitemaps(
    initial_sitemap_urls: List[str],
    client: httpx.AsyncClient,
    domain_config: Dict,
    max_depth=5,
):
    """Processes a list of sitemaps, handling nested sitemap indexes recursively using a queue."""
    sitemap_queue = deque([(url, 0) for url in initial_sitemap_urls])
    processed_count = 0

    while sitemap_queue:
        current_url, current_depth = sitemap_queue.popleft()
        processed_count += 1

        # Check depth limit
        if current_depth > max_depth:
            logger.warning(
                f"[{domain_config['base_domain_netloc']}] Reached max sitemap recursion depth ({max_depth}) at {current_url}, stopping branch."
            )
            continue

        nested_urls = await process_single_sitemap(current_url, client, domain_config)

        for nested_url in nested_urls:
            norm_nested = normalize_url(nested_url, domain_config["base_domain_netloc"])
            if (
                norm_nested
                and is_same_domain(norm_nested, domain_config["base_domain_netloc"])
                and norm_nested not in domain_config["processed_sitemaps"]
            ):
                sitemap_queue.append((norm_nested, current_depth + 1))

    logger.info(
        f"[{domain_config['base_domain_netloc']}] Finished processing {processed_count} sitemap files/indexes."
    )


async def crawl_page(
    url: str,
    client: httpx.AsyncClient,
    domain_config: Dict,
    semaphore: asyncio.Semaphore,
):
    """
    Fetches, parses a single page URL, extracts links, identifies product/category URLs,
    and adds new valid URLs to the crawl queue or product list.
    Operates under the domain's semaphore.
    """
    base_domain_netloc = domain_config["base_domain_netloc"]
    robots_parser = domain_config["robots_parser"]
    user_agent = domain_config["user_agent"]
    discovered_links = domain_config["discovered_links"]
    product_urls = domain_config["product_urls"]
    crawl_queue = domain_config["crawl_queue"]

    crawl_delay = 1.0
    if robots_parser and hasattr(robots_parser, "get_crawl_delay"):
        delay = robots_parser.get_crawl_delay(user_agent)
        if delay is not None:
            crawl_delay = float(delay)
            crawl_delay = max(0.5, crawl_delay)

    async with semaphore:

        if not can_fetch(url, robots_parser, user_agent):
            logger.debug(
                f"[{base_domain_netloc}] Skipping disallowed URL (checked before fetch): {url}"
            )
            return

        logger.debug(f"[{base_domain_netloc}] Crawling: {url}")
        try:
            response = await client.get(url, timeout=25.0)
            final_page_url = str(response.url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type:
                logger.debug(
                    f"[{base_domain_netloc}] Skipping non-HTML content ({content_type}): {final_page_url}"
                )
                await asyncio.sleep(crawl_delay)
                return

            try:
                soup = BeautifulSoup(response.text, "lxml")
            except Exception as parse_err:
                logger.error(
                    f"[{base_domain_netloc}] Failed to parse HTML for {final_page_url}: {parse_err}"
                )
                await asyncio.sleep(crawl_delay)
                return

            # Find and process all links on the page
            links_found_on_page = 0
            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                if (
                    not href
                    or href.startswith("#")
                    or href.lower().startswith("javascript:")
                ):
                    continue  # Skip empty, fragment, or javascript links

                # Resolve relative URLs using the FINAL page URL as base
                resolved_url = urljoin(final_page_url, href)
                normalized_url = normalize_url(resolved_url, base_domain_netloc)

                # Process only valid, same-domain URLs
                if normalized_url and is_same_domain(
                    normalized_url, base_domain_netloc
                ):
                    # Check if we should process this link (not discovered and allowed)
                    if normalized_url not in discovered_links:
                        if can_fetch(normalized_url, robots_parser, user_agent):
                            discovered_links.add(
                                normalized_url
                            )  # Add to discovered *now*
                            links_found_on_page += 1
                            if is_product_url(normalized_url):
                                product_urls.add(normalized_url)
                                logger.debug(
                                    f"[{base_domain_netloc}] Found product link: {normalized_url}"
                                )
                            elif is_category_url(normalized_url):
                                crawl_queue.append(
                                    normalized_url
                                )  # Add to queue for further crawling
                                logger.debug(
                                    f"[{base_domain_netloc}] Found category/crawl link: {normalized_url}"
                                )

            logger.debug(
                f"[{base_domain_netloc}] Found {links_found_on_page} new processable links on: {final_page_url}"
            )

            await asyncio.sleep(crawl_delay)

        except httpx.RequestError as e:
            logger.warning(f"[{base_domain_netloc}] Network error crawling {url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[{base_domain_netloc}] HTTP error {e.response.status_code} crawling {url}"
            )
        except Exception as e:
            logger.error(
                f"[{base_domain_netloc}] Error processing page {url}: {e}",
                exc_info=False,
            )


async def process_domain(
    domain: str,
    client: httpx.AsyncClient,
    user_agent: str,
    concurrency: int,
    max_pages: int = 1000,
) -> Tuple[str, Set[str]]:
    """
    Orchestrates the crawling process for a single domain.
    Handles robots.txt, sitemaps, and manages the page crawling loop with concurrency.
    """
    logger.info(f"[{domain}] Starting processing...")
    initial_domain_netloc = urlparse(domain).netloc

    # --- Domain Specific State ---
    domain_config = {
        "base_domain": domain,
        "base_domain_netloc": initial_domain_netloc,
        "user_agent": user_agent,
        "product_urls": set(),
        "discovered_links": set(),
        "crawl_queue": deque(),
        "robots_parser": None,
        "processed_sitemaps": set(),
    }
    pages_processed_count = 0

    robots_url = get_robots_url(domain)
    robots_parser, final_robots_url = await fetch_and_parse_robots(robots_url, client)
    domain_config["robots_parser"] = robots_parser

    if final_robots_url:
        try:
            final_netloc = urlparse(final_robots_url).netloc
            if final_netloc and final_netloc.lower() != initial_domain_netloc.lower():
                logger.info(
                    f"[{domain}] Domain redirected. Updating netloc from '{initial_domain_netloc}' to '{final_netloc}'."
                )
                domain_config["base_domain_netloc"] = final_netloc.lower()
        except Exception as e:
            logger.warning(
                f"[{domain}] Could not parse final robots URL '{final_robots_url}' to update netloc: {e}"
            )

    current_netloc = domain_config["base_domain_netloc"]
    if not can_fetch(domain, robots_parser, user_agent):
        logger.warning(
            f"[{current_netloc}] Crawling disallowed by robots.txt for entry URL {domain}. Skipping."
        )
        return domain, domain_config["product_urls"]

    # Add base domain to discovered links
    normalized_base = normalize_url(domain, current_netloc)
    if normalized_base:
        domain_config["discovered_links"].add(normalized_base)

    sitemap_urls_from_robots = get_sitemap_urls(robots_parser)
    valid_sitemap_urls = []
    if sitemap_urls_from_robots:
        valid_sitemap_urls = [
            s_url
            for s_url in sitemap_urls_from_robots
            if normalize_url(s_url, current_netloc)
            and is_same_domain(normalize_url(s_url, current_netloc), current_netloc)
        ]
        logger.info(
            f"[{current_netloc}] Found {len(valid_sitemap_urls)} valid sitemap(s) in robots.txt."
        )

    if valid_sitemap_urls:
        await process_sitemaps(valid_sitemap_urls, client, domain_config)
    else:
        logger.info(
            f"[{current_netloc}] No valid sitemaps found in robots.txt. Attempting common location."
        )
        common_sitemap = urljoin(f"https://{current_netloc}", "/sitemap.xml")
        if can_fetch(common_sitemap, robots_parser, user_agent):
            await process_sitemaps([common_sitemap], client, domain_config)

    if not domain_config["crawl_queue"] and normalized_base:
        if (
            normalized_base not in domain_config["discovered_links"]
        ):  # Should already be added, but check
            if can_fetch(normalized_base, robots_parser, user_agent):
                domain_config["discovered_links"].add(
                    normalized_base
                )  # Ensure it's marked discovered
                domain_config["crawl_queue"].append(normalized_base)
                logger.info(
                    f"[{current_netloc}] Seeding crawl queue with base domain: {normalized_base}"
                )
        elif normalized_base in domain_config["discovered_links"] and not any(
            is_category_url(u) or is_product_url(u)
            for u in domain_config["crawl_queue"]
        ):
            # If base was discovered via sitemap but wasn't category/product, maybe still crawl it?
            if can_fetch(normalized_base, robots_parser, user_agent):
                domain_config["crawl_queue"].appendleft(normalized_base)  # Add to front
                logger.info(
                    f"[{current_netloc}] Adding base domain to front of queue for crawling: {normalized_base}"
                )

    logger.info(
        f"[{current_netloc}] Starting crawl phase. Initial queue size: {len(domain_config['crawl_queue'])}. Max pages: {max_pages}"
    )

    semaphore = asyncio.Semaphore(concurrency)  # Concurrency control for this domain
    active_tasks: Set[asyncio.Task] = set()

    try:
        while domain_config["crawl_queue"] and pages_processed_count < max_pages:
            # Spawn new tasks concurrently up to the semaphore limit
            while len(active_tasks) < concurrency and domain_config["crawl_queue"]:
                url_to_crawl = domain_config["crawl_queue"].popleft()
                # Create and track task
                task = asyncio.create_task(
                    crawl_page(url_to_crawl, client, domain_config, semaphore)
                )
                active_tasks.add(task)
                pages_processed_count += 1
                task.add_done_callback(active_tasks.discard)

            if not active_tasks:
                break
            done, pending = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Log progress periodically
            if pages_processed_count % 100 == 0:  # Log every 100 pages
                logger.info(
                    f"[{current_netloc}] Processed ~{pages_processed_count} pages. Queue: {len(domain_config['crawl_queue'])}. Products: {len(domain_config['product_urls'])}"
                )

    except asyncio.CancelledError:
        logger.info(f"[{current_netloc}] Crawl task cancelled.")
    finally:
        # Cleanup: Cancel remaining tasks if loop exited early or was cancelled
        if active_tasks:
            if pages_processed_count >= max_pages:
                logger.info(
                    f"[{current_netloc}] Reached page limit ({max_pages}). Cancelling {len(active_tasks)} outstanding tasks."
                )
            else:
                logger.info(
                    f"[{current_netloc}] Crawl finished/interrupted. Cancelling {len(active_tasks)} outstanding tasks."
                )

            for task in active_tasks:
                task.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)

    logger.info(
        f"[{current_netloc}] Finished crawl phase. Total pages processed: {pages_processed_count}. Total products found: {len(domain_config['product_urls'])}."
    )

    # Return the original domain key passed in, and the final set of product URLs
    return domain, domain_config["product_urls"]


async def main(
    domains: List[str], output_file: str, concurrency: int, max_pages_per_domain: int
):
    """
    Main asynchronous function to initialize and coordinate the crawling process for multiple domains.
    """
    start_time = asyncio.get_event_loop().time()
    logger.info(
        f"Starting crawler at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )  # Use datetime
    logger.info(f"Domains to crawl: {domains}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Concurrency per domain: {concurrency}")
    logger.info(f"Max pages per domain: {max_pages_per_domain}")

    # Configure httpx client with appropriate headers, timeouts, and limits
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # Increased read timeout for potentially slow pages
    timeouts = httpx.Timeout(10.0, read=45.0, write=10.0, pool=5.0)
    # Global limits should accommodate the total potential concurrency
    total_concurrency = concurrency * len(domains)
    limits = httpx.Limits(
        max_connections=total_concurrency + 50,
        max_keepalive_connections=concurrency + 10,
    )  # Generous limits

    async with httpx.AsyncClient(
        http2=True,
        follow_redirects=True,
        headers=headers,
        timeout=timeouts,
        limits=limits,
    ) as client:
        tasks = []
        initial_domains = {}  # Map base_domain back to original input string

        # Create tasks for each domain
        for i, domain_url_input in enumerate(domains):
            domain_url = domain_url_input
            if not domain_url.startswith(("http://", "https://")):
                domain_url = f"https://{domain_url}"  # Default to https
            try:
                parsed_uri = urlparse(domain_url)
                # Use lower() for consistent key mapping
                base_domain = (
                    f"{parsed_uri.scheme.lower()}://{parsed_uri.netloc.lower()}"
                )
                initial_domains[base_domain] = (
                    domain_url_input  # Map clean base domain to original
                )

                # Stagger task creation slightly
                if i > 0:
                    await asyncio.sleep(0.15)  # Slightly longer delay

                tasks.append(
                    asyncio.create_task(
                        process_domain(
                            base_domain,
                            client,
                            USER_AGENT,
                            concurrency,
                            max_pages_per_domain,
                        )
                    )
                )
            except Exception as e:
                logger.error(
                    f"Failed to create task for domain '{domain_url_input}': {e}"
                )

        domain_results_list = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {}
        for result in domain_results_list:
            original_domain_key = "UnknownDomain"
            try:
                if isinstance(result, Exception):
                    logger.error(
                        f"Task for a domain failed with exception: {result}",
                        exc_info=True,
                    )
                elif result and isinstance(result, tuple) and len(result) == 2:
                    domain_key, product_urls = (
                        result  # domain_key is the initial base_domain passed
                    )
                    original_domain_key = initial_domains.get(domain_key, domain_key)
                    if product_urls:
                        final_results[original_domain_key] = sorted(list(product_urls))
                    else:
                        logger.info(
                            f"No product URLs found for domain: {original_domain_key}"
                        )
                else:
                    logger.warning(
                        f"Unexpected result format received from a domain task: {result}"
                    )
            except Exception as e:
                logger.error(
                    f"Error processing result for domain {original_domain_key}: {e}",
                    exc_info=True,
                )

    if final_results:
        logger.info(
            f"Writing {sum(len(v) for v in final_results.values())} product URLs from {len(final_results)} domains to {output_file}"
        )
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote results to {output_file}")
        except IOError as e:
            logger.error(f"Error writing results to {output_file}: {e}")
    else:
        logger.info("No product URLs identified for any domain after full crawl.")

    end_time = asyncio.get_event_loop().time()
    logger.info(f"Crawler finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    import datetime

    parser = argparse.ArgumentParser(
        description="Asynchronous E-commerce Product URL Crawler using httpx and asyncio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help
    )
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        required=True,
        help="List of domain names to crawl (e.g., virgio.com tatacliq.com)",
    )
    parser.add_argument(
        "-o", "--output", default="product_urls.json", help="Output JSON file path"
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests *per domain*",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Maximum number of pages to crawl per domain (0 for unlimited)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Handle max_pages=0 meaning unlimited (or a very large number)
    max_pages_limit = args.max_pages if args.max_pages > 0 else float("inf")

    # Run the main asynchronous function
    try:
        asyncio.run(
            main(args.domains, args.output, args.concurrency, int(max_pages_limit))
        )
    except KeyboardInterrupt:
        logger.info("Crawler interrupted by user.")
    except Exception as e:
        logger.exception(
            f"An unexpected critical error occurred in main execution: {e}"
        )
