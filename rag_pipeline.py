import os
import re
import json
import requests
import pandas as pd
from urllib.parse import urlparse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("PERSONAL")

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

def load_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found.")
        return ""
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return ""

def get_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain
    except Exception:
        return "unknown_domain"

def clean_text_content(text):
    if text is None:
        return "Tidak ada konten yang tersedia."
    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sanitize_content(content):
    if content is None or content.strip() == "" or content.strip() == "Tidak ada konten yang tersedia.":
        return "Tidak ada konten yang tersedia."
    content = clean_text_content(content)
    content = content.replace("{", "{{").replace("}", "}}")
    if len(content) > 10000:
        content = content[:10000] + "..."
    return content

def export_final_prompt(user_input, evidence_sources, system_prompt, output_path="./output/final_prompt.txt"):
    user_prompt = f"Klaim Pengguna: {user_input}\n"
    if evidence_sources:
        user_prompt += "\\nHasil Pencarian:\\n"
        for i, source in enumerate(evidence_sources):
            content = source.get('extracted_content') or source.get('content') or "Tidak ada konten yang tersedia."
            content = sanitize_content(content)
            user_prompt += f"{i+1}. Judul: {source.get('title')}, URL: {source.get('url')}, Domain: {source.get('domain')}, Score: {source.get('score')}\\n   Konten: {content}\\n\\n"
    full_prompt = f"System Prompt:\\n{system_prompt}\\n\\nUser Prompt:\\n{user_prompt}"
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        return full_prompt
    except Exception as e:
        print(f"Error exporting prompt: {str(e)}")
        return None

def load_json_file(json_filename="domain/15domain.json"):
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {json_filename} file not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filename}. Make sure it's a valid JSON file.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while loading JSON: {str(e)}")
        return None

def generate_queries(userInput):
    if not userInput or len(userInput.strip()) == 0:
        return ["No valid input provided"], "Error: Empty input"
    prompt_query = load_text_file("prompts/query_gen-3questions_one-shot.txt")
    query_prompt = ChatPromptTemplate.from_template(prompt_query)
    query_chain = query_prompt | llm | StrOutputParser()
    try:
        query_text = query_chain.invoke({"query": userInput})
        # lines = query_text.strip().split('\\n')
        lines = query_text.strip().split('\n')
        queries = []
        pattern = r'^\s*\d+\.\s*(.+)$'
        for line in lines:
            match = re.match(pattern, line)
            if match:
                query = match.group(1).strip()
                queries.append(query)
        queries = queries[:3]
        if not queries:
            queries = [userInput[:200]]
        return queries, query_text
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [userInput[:200]], f"Error generating queries: {str(e)}"

def search_tavily(queries):
    all_search_results = {"results": []}
    trusted_domains = load_json_file()

    for query in queries:
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": 10,
            "include_domains": trusted_domains,
            "exclude_domains": [],
            "include_answer": False,
            "include_raw_content": True,
            "include_images": False,
            "include_image_descriptions": False
        }
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            search_results = response.json()
            for result in search_results.get('results', []):
                result['source_query'] = query
            all_search_results['results'].extend(search_results.get('results', []))
        except requests.exceptions.Timeout:
            print(f"Timeout error searching Tavily for query '{query}'")
        except requests.exceptions.RequestException as e:
            print(f"Request error searching Tavily for query '{query}': {e}")
        except Exception as e:
            print(f"Error searching Tavily for query '{query}': {e}")
    return all_search_results

def extract_content_from_urls(urls):
    if not urls:
        print("No URLs provided for extraction")
        return {}
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your-key-here":
        print("Error: TAVILY_API_KEY not properly configured for extraction")
        return {}
    url_to_content = {}
    batch_size = 5
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        print(f"Extracting content from batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1} ({len(batch_urls)} URLs)")
        try:
            payload = {
                "urls": batch_urls,
                "include_images": False,
                "extract_depth": "basic"
            }
            response = requests.post(
                "https://api.tavily.com/extract",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                extraction_results = response.json()
                for result in extraction_results.get('results', []):
                    url = result.get('url')
                    content = result.get('raw_content')
                    if url and content:
                        url_to_content[url] = content
                if extraction_results.get('failed_results'):
                    print(f"Warning: {len(extraction_results.get('failed_results'))} URLs failed to extract in batch {i//batch_size + 1}")
            else:
                print(f"Error: Extraction request failed with status code {response.status_code}")
                print(f"Response: {response.text[:200]}...")
        except requests.exceptions.Timeout:
            print(f"Timeout error extracting content for batch {i//batch_size + 1}")
        except requests.exceptions.RequestException as e:
            print(f"Request error extracting content for batch {i//batch_size + 1}: {e}")
        except Exception as e:
            print(f"Error extracting content for batch {i//batch_size + 1}: {str(e)}")
    print(f"Successfully extracted content from {len(url_to_content)}/{len(urls)} URLs")
    return url_to_content

def process_search_results(search_results):
    if not search_results or 'results' not in search_results:
        print("No valid search results provided.")
        return {"results": [], "stats": {"total_initial_results": 0, "total_processed_results": 0, "queries_used": []}}
    sanitized_results = []
    for result in search_results.get('results', []):
        domain = get_domain(result['url'])
        result['domain'] = domain
        if 'content' in result and result['content'] is not None:
            result['content'] = sanitize_content(result['content'])
        result['extracted_content'] = None
        sanitized_results.append(result)
    query_results = {}
    for item in sanitized_results:
        query = item['source_query']
        if query not in query_results:
            query_results[query] = []
        query_results[query].append(item)
    final_results = []
    for query, results in query_results.items():
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        top_results = sorted_results[:5]
        final_results.extend(top_results)
    unique_results = {}
    for item in final_results:
        if item['url'] not in unique_results:
            unique_results[item['url']] = item
    deduped_results = list(unique_results.values())
    urls_to_extract = []
    original_raw_content_map = {item['url']: item.get('raw_content') for item in search_results.get('results', []) if item.get('url')}
    for item in deduped_results:
        url = item.get('url')
        if not url:
            continue
        original_raw_content = original_raw_content_map.get(url)
        if original_raw_content is None or isinstance(original_raw_content, str) and original_raw_content.strip() == "":
            urls_to_extract.append(url)
    url_to_extracted_content = {}
    if urls_to_extract:
        urls_to_extract = list(set(urls_to_extract))
        print(f"Attempting to extract content for {len(urls_to_extract)} results missing raw_content...")
        url_to_extracted_content = extract_content_from_urls(urls_to_extract)
    else:
        print("No results missing raw_content, skipping full content extraction step.")
    for result in deduped_results:
        url = result.get('url')
        if url and url in url_to_extracted_content:
            result['extracted_content'] = sanitize_content(url_to_extracted_content[url])
        if 'raw_content' in result and result['raw_content'] is not None:
            result['raw_content'] = sanitize_content(result['raw_content'])
    processed_results = {
        "results": deduped_results,
        "stats": {
            "total_initial_results": len(search_results.get('results', [])),
            "total_processed_results": len(deduped_results),
            "urls_extracted_count": len(url_to_extracted_content),
            "queries_used": list(query_results.keys())
        }
    }
    return processed_results

def write_initial_results_to_json(initial_results, filename="initial_search_results.json"):
    query_grouped_results = {}
    for result in initial_results.get('results', []):
        query = result.get('source_query', 'Unknown Query')
        if query not in query_grouped_results:
            query_grouped_results[query] = []
        query_grouped_results[query].append(result)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(query_grouped_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing initial results to JSON: {e}")

def write_processed_results_to_json(processed_results, filename="processed_search_results.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing processed results to JSON: {e}")

def format_final_response(user_input, processed_results):
    system_prompt = load_text_file("prompts/system_prompt2.txt")
    user_prompt = f"Klaim yang perlu di-fact check: {user_input}\n"
    evidence_sources = processed_results['results']
    if evidence_sources:
        user_prompt += "\\nSumber-Sumber Bukti:\\n"
        for i, source in enumerate(evidence_sources):
            content_to_use = (
                    sanitize_content(source.get('extracted_content')) or
                    sanitize_content(source.get('raw_content')) or
                    sanitize_content(source.get('content')) or
                    "Tidak ada konten yang tersedia."
            )
            user_prompt += f"{i+1}. Judul: {source.get('title')}\\n   URL: {source.get('url')}\\n   Domain: {source.get('domain')}\\n   Score: {source.get('score')}\\n   Konten: {content_to_use}\\n\\n"
    else:
        user_prompt += "\\nPeringatan: Tidak ada sumber bukti yang ditemukan. Analisis mungkin tidak akurat.\\n"
    export_final_prompt(user_input, evidence_sources, system_prompt)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "evidence_count": len(evidence_sources)
    }

def generate_fact_check_analysis(user_query, processed_results):
    if isinstance(processed_results, dict) and 'results' in processed_results:
        evidence_results = processed_results['results']
    elif isinstance(processed_results, list):
        evidence_results = processed_results
    else:
        evidence_results = []
        print("Warning: Format evidence tidak sesuai yang diharapkan.")
    system_prompt = load_text_file("prompts/system_prompt2.txt")
    user_prompt = f"Klaim yang perlu di-fact check: {user_query}\nSumber-sumber bukti yang tersedia:\n"
    for i, evidence in enumerate(evidence_results, 1):
        if isinstance(evidence, dict):
            title = evidence.get('title', 'Tidak ada judul')
            url = evidence.get('url', 'Tidak ada URL')
            domain = evidence.get('domain', 'Tidak ada domain')
            score = evidence.get('score', 0)
            content = (
                    evidence.get('extracted_content') or
                    evidence.get('raw_content') or
                    evidence.get('content') or
                    "Tidak ada konten yang tersedia."
            )
            content = str(content) if content is not None else "Tidak ada konten yang tersedia."
            content = sanitize_content(content)
        else:
            title = "Format tidak valid"
            url = "N/A"
            domain = "N/A"
            score = 0
            content = str(evidence)[:500]
            content = sanitize_content(content)
        user_prompt += f"""
        BUKTI {i}:
        Judul: {title}
        URL: {url}
        Domain: {domain}
        Skor Relevansi: {score}

        Konten: {content}

        ---
        """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    fact_check_llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    generation_chain = prompt | fact_check_llm | StrOutputParser()
    try:
        analysis = generation_chain.invoke({})
        os.makedirs("./output", exist_ok=True)
        with open("./output/fact_check_analysis.md", "w", encoding="utf-8") as f:
            f.write(analysis)
        full_log_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        with open("./output/full_prompt.txt", "w", encoding="utf-8") as f:
            f.write(f"SYSTEM PROMPT:\\n{system_prompt}\\n\\nUSER PROMPT:\\n{user_prompt}")
        return {
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        error_message = f"Error generating fact check analysis: {str(e)}"
        print(error_message)
        return {
            "analysis": error_message,
            "status": "error"
        }

def process_output(user_input_text):
    results = {}
    # Streamlit will handle the printing, so we can replace print with st.write or st.info
    # For now, keeping print for direct console debugging if needed.

    # Step 1: User Input
    results["user_input"] = user_input_text

    # Step 2: Query Generation
    queries, query_text = generate_queries(user_input_text)
    results["queries"] = queries

    # Step 3: Web Search
    search_results = search_tavily(queries)
    results["search_results_count"] = len(search_results.get('results', []))

    # Step 4: Processing Results and Content Extraction
    processed_results = process_search_results(search_results)
    results["processed_results"] = processed_results
    os.makedirs("./output", exist_ok=True) # Ensure output dir exists before writing
    write_initial_results_to_json(search_results, "output/initial_search_results_grouped.json")
    write_processed_results_to_json(processed_results, "output/processed_search_results.json")

    # Step 5: Final Response
    final_response_data = format_final_response(user_input_text, processed_results)
    results["evidence_count"] = final_response_data['evidence_count']

    # Step 6: Generation - Fact Check Analysis
    fact_check_result = generate_fact_check_analysis(user_input_text, processed_results)
    results["fact_check_analysis"] = fact_check_result

    return results