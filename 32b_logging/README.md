# 32B Logging

`--logging` mode writes human-readable traces here.

Typical layout:

~~~text
32b_logging/
  active_run.env
  latest -> runs/<run_id>
  runs/<run_id>/
    server/
      run_server.env
      vllm.log
      vllm.pid
    inference/
      logging_manifest.json
      events.jsonl
    samples/<data_id>/
      summary.md
      sample_meta.json
      events.jsonl
      stages/
      llm_calls/
~~~

`summary.md` is the fastest place to inspect a sample end-to-end.
