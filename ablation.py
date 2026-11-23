"""Measure cumulative single-component variants; generation requires --generate."""
import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


VARIANTS = [
    ("B0", "Naive baseline"),
    ("V1", "Preserve table labels"),
    ("V2", "Remove navigation and non-content HTML"),
    ("V3", "Replace fixed-size chunks with document blocks and table rows"),
    ("V4", "Prepend company and section context"),
    ("V5", "Use alphabetic token boundaries"),
    ("V6", "Remove stop words"),
    ("V7", "Normalize financial synonyms"),
    ("V8", "Remove parsed entity clauses from ranking query"),
    ("V9", "Add query-token coverage score"),
    ("V10", "Add table preference"),
    ("V11", "Add exact-report-year preference"),
    ("V12", "Enable company/type/year candidate filtering"),
    ("V13", "Allocate retrieval slots per company"),
    ("V14", "Enable evidence-coverage abstention"),
]


def configuration(variant):
    return dict(table_text=variant >= 1, clean_html=variant >= 2,
                block_chunks=variant >= 3, context_headers=variant >= 4,
                alphabetic_tokens=variant >= 5, stop_words=variant >= 6,
                synonyms=variant >= 7, query_preprocessing=variant >= 8,
                coverage_boost=variant >= 9, table_boost=variant >= 10,
                year_boost=variant >= 11, metadata_filter=variant >= 12,
                company_allocation=variant >= 13, abstention=variant >= 14)


def tokenizer_for(variant):
    def tokenize(text):
        words = re.findall(r"[a-z]+", text.lower()) if variant >= 5 else text.lower().split()
        if variant >= 6:
            words = [word for word in words if word not in hw4.STOP_WORDS]
        if variant >= 7:
            words = [hw4.ALIASES.get(word, word) for word in words if hw4.ALIASES.get(word, word)]
        return words
    return tokenize


def make_chunks(corpus, variant):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = []
    for name, html in corpus.items():
        soup = BeautifulSoup(html, "html.parser")
        metadata = hw4.metadata_from_html(name, soup)
        if variant >= 3:
            for chunk in hw4.chunk_document(name, html):
                if variant == 3:
                    chunk = Document(page_content=chunk.page_content.split("\n", 1)[1],
                                     metadata=chunk.metadata)
                chunks.append(chunk)
            continue
        if variant >= 2:
            for node in soup.find_all(["nav", "style", "script", "head"]):
                node.decompose()
        if variant >= 1:
            for table in soup.find_all("table"):
                table.replace_with(soup.new_string(hw4.table_to_text(str(table))))
        chunks.extend(Document(page_content=piece, metadata=metadata)
                      for piece in splitter.split_text(soup.get_text(" ", strip=True)))
    return chunks


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(folder, output, generate):
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    gold = hw4.load_json(folder / "ground_truth.json")
    if not gold:
        raise ValueError("Evaluation requires ground_truth.json; retrieval never reads it.")
    output.mkdir(parents=True, exist_ok=True)
    configurations = {label: configuration(variant) for variant, (label, _) in enumerate(VARIANTS)}
    previous = configurations["B0"]
    for label, _ in VARIANTS[1:]:
        current = configurations[label]
        assert sum(current[key] != previous[key] for key in current) == 1
        previous = current
    (output / "configurations.json").write_text(json.dumps(configurations, indent=2) + "\n")
    summary = []
    for variant, (label, change) in enumerate(VARIANTS):
        started = time.perf_counter()
        chunks = make_chunks(corpus, variant)
        tokenizer = tokenizer_for(variant)
        tokenized = [tokenizer(chunk.page_content) for chunk in chunks]
        index = {"chunks": chunks, "bm25": BM25Okapi(tokenized),
                 "tokens": [set(words) for words in tokenized]}
        retrieved = {}
        for qid, question in questions.items():
            retrieved[qid] = hw4.retrieve_config(index, question, filtering=variant >= 12,
                balanced=variant >= 13, abstention=variant >= 14, tokenizer=tokenizer,
                preprocess=variant >= 8, coverage_weight=12 if variant >= 9 else 0,
                table_weight=3 if variant >= 10 else 0, year_weight=1 if variant >= 11 else 0)
            if variant == 14:
                expected = hw4.retrieve(index, question)
                assert [(chunk.page_content, chunk.metadata) for chunk in retrieved[qid]] == [
                    (chunk.page_content, chunk.metadata) for chunk in expected]
        retrieval_seconds = time.perf_counter() - started
        usage_before = hw4.GENERATOR_USAGE.summary()
        answers = {}
        if generate:
            with ThreadPoolExecutor(max_workers=8) as pool:
                values = pool.map(lambda qid: hw4.generate(questions[qid], retrieved[qid]), questions)
                answers = dict(zip(questions, values))
        rows = []
        for qid, question in questions.items():
            result = retrieved[qid]
            rows.append(dict(id=qid, category=gold[qid]["category"], question=question,
                parsed=json.dumps(hw4.parse_query(question)) if variant >= 8 else "",
                retrieved=";".join(sorted({chunk.metadata["source"] for chunk in result})),
                chunk_count=len(result), retrieval_hit=hw4.retrieval_hit(result, gold[qid])
                if gold[qid]["evidence"] else "", abstained=not result,
                answer=answers.get(qid, ""), correct=hw4.answer_correct(answers[qid], gold[qid])
                if generate else "", context=hw4.format_context(result)))
        answerable = [row for row in rows if gold[row["id"]]["evidence"]]
        unanswerable = [row for row in rows if not gold[row["id"]]["evidence"]]
        usage_after = hw4.GENERATOR_USAGE.summary()
        summary.append(dict(variant=label, change=change, chunks=len(chunks),
            recall_at_5=sum(row["retrieval_hit"] for row in answerable) / len(answerable),
            kept=sum(not row["abstained"] for row in answerable) / len(answerable),
            rejected=sum(row["abstained"] for row in unanswerable) / len(unanswerable),
            accuracy=sum(row["correct"] for row in rows) / len(rows) if generate else "",
            retrieval_seconds=round(retrieval_seconds, 3),
            total_seconds=round(time.perf_counter() - started, 3),
            generator_input_tokens=usage_after["llm_input_tokens"] - usage_before["llm_input_tokens"],
            generator_output_tokens=usage_after["llm_output_tokens"] - usage_before["llm_output_tokens"]))
        write_csv(output / f"{label}.csv", rows)
        write_csv(output / "summary.csv", summary)
        print(json.dumps(summary[-1]), flush=True)
    checks = {}
    tables = hw4.load_json(folder / "tables.json")
    checks["table_scores"] = [hw4.table_score(hw4.table_to_text(table["html"]), table) for table in tables]
    metadata = hw4.load_json(folder / "docs.json")
    checks["document_problems"] = {name: hw4.document_problems(hw4.chunk_document(name, html), None,
        metadata[name], name) for name, html in corpus.items()}
    checks["parse_scores"] = {qid: hw4.parse_score(hw4.parse_query(question), gold[qid])
                              for qid, question in questions.items()}
    checks["generation_run"] = generate
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/single-factor-offline"))
    parser.add_argument("--generate", action="store_true", help="Call the original fixed generator for every variant")
    args = parser.parse_args()
    run(args.data_folder, args.output_dir, args.generate)
