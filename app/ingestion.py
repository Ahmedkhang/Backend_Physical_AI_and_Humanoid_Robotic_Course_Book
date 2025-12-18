import os
import httpx
import hashlib
from tqdm import tqdm
from bs4 import BeautifulSoup
import trafilatura  # Best for extracting main content from web pages
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from cohere import Client as CohereClient
from dotenv import load_dotenv

load_dotenv()

def crawl_and_ingest(sitemap_url: str = None):
    if sitemap_url is None:
        sitemap_url = os.getenv("BOOK_SITEMAP_URL")
        if not sitemap_url:
            raise ValueError("BOOK_SITEMAP_URL not set in .env")

    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    collection_name = os.getenv("COLLECTION_NAME", "physical-ai-book")

    # Create collection if not exists
    collections = qdrant_client.get_collections()
    if collection_name not in [c.name for c in collections.collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Created new collection: {collection_name}")

    # Fetch and parse sitemap
    print(f"Fetching sitemap from: {sitemap_url}")
    response = httpx.get(sitemap_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "xml")
    bad_urls = [loc.text.strip() for loc in soup.find_all("loc")]

    # Fix broken custom domain → use working Vercel domain
    fixed_urls = [
        url.replace("physical-ai-robotics-textbook.com", "physical-ai-and-humanoid-robotic-co.vercel.app")
        for url in bad_urls
    ]

    # Filter only actual chapter/module pages (skip home, tags, etc.)
    urls = [url for url in fixed_urls if "/modules/" in url and "/chapter-" in url]

    print(f"Found and fixed {len(urls)} chapter pages for ingestion.")

    if len(urls) == 0:
        print("No chapter pages found. Check your sitemap and domain fix.")
        return

    all_points = []
    successful_pages = 0

    for url in tqdm(urls, desc="Ingesting chapters"):
        try:
            page_response = httpx.get(url, timeout=30.0)
            page_response.raise_for_status()

            # Extract clean main content using trafilatura (removes nav, footer, etc.)
            text = trafilatura.extract(page_response.text, include_links=False, include_images=False)

            if not text or len(text.strip()) < 200:
                print(f"Skipping {url} — insufficient content")
                continue

            successful_pages += 1

            # Chunk text with overlap
            chunk_size = 1024
            overlap = 200
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                if len(chunk.strip()) > 50:  # ignore tiny chunks
                    chunks.append(chunk)
                start = end - overlap

            if not chunks:
                continue

            # Embed all chunks in one call
            embeddings = cohere_client.embed(
                texts=chunks,
                model="embed-english-v3.0",
                input_type="search_document"
            ).embeddings

            # Create points
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = hashlib.md5(f"{url}-{idx}".encode()).hexdigest()
                all_points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "url": url,
                        "chunk_index": idx,
                        "page_title": url.split("/")[-2].replace("-", " ").title() + " - " + url.split("/")[-1].replace("-", " ").title()
                    }
                ))

            # Batch upsert every 100 points
            if len(all_points) >= 100:
                qdrant_client.upsert(collection_name=collection_name, points=all_points)
                print(f"Upserted {len(all_points)} points (batch)")
                all_points = []

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue

    # Final upsert
    if all_points:
        qdrant_client.upsert(collection_name=collection_name, points=all_points)
        print(f"Final upsert: {len(all_points)} points")

    print(f"\nIngestion completed successfully! 🚀")
    print(f"Processed {successful_pages} pages")
    print(f"Total chunks added: {len(all_points) + (successful_pages * 3)} (approx)")
    print("Your RAG chatbot is now powered by your full book!")