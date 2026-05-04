import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_slug(value, fallback="item", max_len=80):
    text = str(value or "")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.").lower()
    if not slug:
        slug = fallback
    return slug[:max_len]


def _code_block(text, language="text"):
    return f"~~~{language}\n{text}\n~~~\n"


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


class NullTraceLogger:
    enabled = False
    run_dir = None

    def log_run_manifest(self, payload):
        return None

    def log_run_event(self, event, payload=None):
        return None

    def start_sample(self, data_id, payload):
        return None

    def log_stage(self, data_id, stage, payload, title=None, include_in_summary=True):
        return None

    def log_llm_call(self, data_id, component, prompt, response, metadata=None):
        return None

    def log_error(self, data_id, payload):
        return None


class StructRAGTraceLogger:
    enabled = True

    def __init__(self, run_dir, run_id=None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.inference_dir = self.run_dir / "inference"
        self.samples_dir = self.run_dir / "samples"
        self._lock = threading.Lock()
        self._seq = 0

        self.inference_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def _next_seq(self):
        with self._lock:
            self._seq += 1
            return self._seq

    def _append_jsonl(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_text(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _append_text(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _sample_dir(self, data_id):
        sample_slug = _safe_slug(data_id, fallback="unknown-sample")
        sample_dir = self.samples_dir / sample_slug
        (sample_dir / "llm_calls").mkdir(parents=True, exist_ok=True)
        (sample_dir / "stages").mkdir(parents=True, exist_ok=True)
        return sample_dir

    def _render_payload_markdown(self, payload):
        if isinstance(payload, dict):
            lines = []
            for key, value in payload.items():
                lines.append(f"## {key}\n")
                language = "text" if isinstance(value, str) else "json"
                lines.append(_code_block(_stringify(value), language))
            return "\n".join(lines).rstrip() + "\n"
        language = "text" if isinstance(payload, str) else "json"
        return _code_block(_stringify(payload), language)

    def _summary_path(self, data_id):
        return self._sample_dir(data_id) / "summary.md"

    def log_run_manifest(self, payload):
        manifest_path = self.inference_dir / "logging_manifest.json"
        self._write_text(
            manifest_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def log_run_event(self, event, payload=None):
        self._append_jsonl(
            self.inference_dir / "events.jsonl",
            {
                "timestamp": _utc_now(),
                "event": event,
                "payload": payload or {},
            },
        )

    def start_sample(self, data_id, payload):
        sample_dir = self._sample_dir(data_id)
        meta_path = sample_dir / "sample_meta.json"
        self._write_text(meta_path, json.dumps(payload, ensure_ascii=False, indent=2))

        summary_path = self._summary_path(data_id)
        if not summary_path.exists():
            header = [
                f"# Sample {data_id}",
                "",
                f"- run_id: {self.run_id or 'unknown'}",
                f"- created_at: {_utc_now()}",
                "",
            ]
            self._write_text(summary_path, "\n".join(header))

        self._append_text(
            summary_path,
            "## Input Metadata\n\n" + self._render_payload_markdown(payload) + "\n",
        )

    def log_stage(self, data_id, stage, payload, title=None, include_in_summary=True):
        seq = self._next_seq()
        sample_dir = self._sample_dir(data_id)
        stage_slug = _safe_slug(stage, fallback="stage")
        stage_title = title or stage
        relative_path = f"stages/{seq:03d}_{stage_slug}.md"
        stage_path = sample_dir / relative_path

        document = [f"# {stage_title}", "", self._render_payload_markdown(payload)]
        self._write_text(stage_path, "\n".join(document).rstrip() + "\n")

        self._append_jsonl(
            sample_dir / "events.jsonl",
            {
                "timestamp": _utc_now(),
                "type": "stage",
                "stage": stage,
                "title": stage_title,
                "path": relative_path,
            },
        )

        if include_in_summary:
            self._append_text(
                self._summary_path(data_id),
                f"## {stage_title}\n\n{self._render_payload_markdown(payload)}\n",
            )

    def log_llm_call(self, data_id, component, prompt, response, metadata=None):
        seq = self._next_seq()
        sample_dir = self._sample_dir(data_id)
        component_slug = _safe_slug(component, fallback="llm-call")
        relative_path = f"llm_calls/{seq:03d}_{component_slug}.md"
        output_path = sample_dir / relative_path

        metadata = metadata or {}
        document_lines = [
            f"# LLM Call: {component}",
            "",
            "## Metadata",
            "",
            _code_block(_stringify(metadata), "json"),
            "## Prompt",
            "",
            _code_block(_stringify(prompt), "text"),
            "## Response",
            "",
            _code_block(_stringify(response), "text"),
        ]
        self._write_text(output_path, "\n".join(document_lines).rstrip() + "\n")

        self._append_jsonl(
            sample_dir / "events.jsonl",
            {
                "timestamp": _utc_now(),
                "type": "llm_call",
                "component": component,
                "path": relative_path,
                "metadata": metadata,
            },
        )

        self._append_text(
            self._summary_path(data_id),
            (
                "## LLM Call Trace\n\n"
                f"- component: `{component}`\n"
                f"- detail_file: `{relative_path}`\n\n"
            ),
        )

    def log_error(self, data_id, payload):
        self.log_stage(
            data_id=data_id,
            stage="error",
            payload=payload,
            title="Error",
            include_in_summary=True,
        )


def build_trace_logger_from_env():
    if os.environ.get("STRUCTRAG_LOGGING", "0") != "1":
        return NullTraceLogger()

    logging_dir = os.environ.get("STRUCTRAG_LOGGING_DIR")
    if not logging_dir:
        return NullTraceLogger()

    return StructRAGTraceLogger(
        run_dir=logging_dir,
        run_id=os.environ.get("STRUCTRAG_LOGGING_RUN_ID"),
    )
