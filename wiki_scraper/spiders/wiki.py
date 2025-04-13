import random
from datetime import datetime
from urllib.parse import urljoin
import re
from scrapy_splash import SplashRequest
from pathlib import Path
from scrapy import Spider
import json
from wiki_scraper.settings import  CRAWLER_LINKS_LOWER_THRESHOLD, CRAWLER_LINKS_UPPER_THRESHOLD, RESULTS_DIR, REQUEST_WAIT



HYPERLINKS_SELECTOR = "#bodyContent p>a"
def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H-%M")


def _get_links(response) -> set[str]:
    links = set()
    links_limit = random.randint(
        CRAWLER_LINKS_LOWER_THRESHOLD,
        CRAWLER_LINKS_UPPER_THRESHOLD
    )
    links_iter = 0

    for href in response.css("#bodyContent a::attr(href)").getall():
        if href.startswith("/wiki/") and not re.search(r':|#', href):
            full_url = urljoin(response.url, href)
            links.add(full_url)
            links_iter += 1
        if links_iter >= links_limit:
            break
    return links



class WikiScraper(Spider):
    name = "wiki_graph"
    allowed_domains = ["en.wikipedia.org", "127.0.0.1"]
    start_urls = ["https://en.wikipedia.org/wiki/Web_scraping"]

    graph = {}
    visited = set()
    request_counter: int = 0
    
    def _send_request(self, url: str, wait: float = REQUEST_WAIT, **meta_kwargs,):
        self.request_counter += 1
        return SplashRequest(url=url, callback=self.parse, args={"wait": wait}, meta=meta_kwargs)

    def _log_state(self, page: str, depth: int) -> None:
        self.logger.info(
            "[Request %s][Depth %s]Current page: %s",
            self.request_counter,
            depth,
            page
        )

    def start_requests(self):
        for url in self.start_urls:
            yield self._send_request(url)

    def parse(self, response):
        current_page = response.url
        depth = response.meta.get('depth', 0)
        self._log_state(current_page, depth)

        if current_page in self.visited:
            return
        self.visited.add(current_page)

        links = _get_links(response)

        self.graph[current_page] = list(links)

        self.logger.info("Links to serach %s, %s", len(links), links)
        for link in links:
            yield self._send_request(link, depth=depth+1)
            

    def closed(self, reason):
        output_file = RESULTS_DIR / ("wiki_graph" + get_current_time_string())
        with output_file.open("w") as f:
            json.dump(self.graph, f, indent=2)
        
