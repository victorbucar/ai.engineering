"""Manual mode: chat with the bank assistant through the FULL guardrail stack.

Every turn goes through three layers, and the trace shows what each one decided:

  L1  input guardrails (this folder, regex)     length, prompt injection, PII, harmful topic
  L2  NeMo Guardrails + Claude (nemo/)          self check input -> dialog rails -> answer
                                                -> self check output
  L3  output guardrails (this folder)           OpenAI moderation, system-prompt leak,
                                                relevance (advisory), PII scrubbing

Run:  python chat.py
Commands:  /trace  toggle the per-layer trace     /reset  start a new conversation
           /stats  monitoring dashboard           /quit   exit (or an empty line)
Needs ANTHROPIC_API_KEY (Claude, via NeMo) and OPENAI_API_KEY (moderation).
"""
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nemo"))

from nemo_bot import CONFIG_DIR, DISCLOSURE_RE, build_rails  # noqa: E402  (also sets the NeMo framework)
from guardrail_types import GuardrailReport  # noqa: E402
from guardrail_pipeline import GuardrailPipeline  # noqa: E402
from guardrail_monitor import GuardrailMonitor  # noqa: E402

INPUT_BLOCKED_REPLY = "I cannot process this request. Please rephrase your question."
OUTPUT_BLOCKED_REPLY = "I apologize, but I cannot provide that response. Let me help you differently."

# Relevance is word overlap with the question. NeMo's dialog rails already own topic control,
# and their fixed redirects ("I can only help with banking...") share no words with an
# off-topic question, so in this mode relevance is reported but never blocks.
ADVISORY_CHECKS = {"relevance_check", "relevance"}


def load_system_prompt():
    """The NeMo 'general' instructions, used by L3 to detect the bot echoing its own prompt."""
    with open(os.path.join(CONFIG_DIR, "config.yml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return "\n".join(i["content"] for i in config.get("instructions", []) if i.get("type") == "general")


def fmt_checks(results):
    parts = []
    for r in results:
        name = r.category.replace("_detection", "").replace("_check", "").replace("_filter", "")
        if r.category in ADVISORY_CHECKS:
            state = f"{r.confidence:.2f} (advisory)"
        elif r.category == "pii_scrubbing":
            state = "scrubbed" if not r.passed else "nothing to scrub"
        else:
            state = "ok" if r.passed else f"BLOCK {r.confidence:.2f}"
        parts.append(f"{name} {state}")
    return " | ".join(parts)


class FullStackChat:
    def __init__(self):
        print("Loading NeMo Guardrails (first run downloads the embedding model)...")
        self.rails = build_rails()
        self.pipeline = GuardrailPipeline(system_prompt=load_system_prompt())
        self.monitor = GuardrailMonitor()
        self.history = []
        self.trace = True

    def ask_nemo(self, user_input):
        result = self.rails.generate(
            messages=self.history + [{"role": "user", "content": user_input}],
            options={"log": {"activated_rails": True}},
        )
        answer = result.response[0]["content"] if isinstance(result.response, list) else str(result.response)
        rails = result.log.activated_rails
        steps = " -> ".join(f"{r.name}{' STOP' if r.stop else ''}" for r in rails) or "n/a"
        # An input/output rail that stops means NeMo replaced the answer with its refusal.
        stopped_by = next((r.name for r in rails if r.stop and r.type in ("input", "output")), None)
        return answer, steps, stopped_by

    def turn(self, user_input):
        start = time.time()
        report = GuardrailReport()
        lines = []

        # L1: regex input guardrails. A blocked turn never reaches Claude or the history.
        report.input_results = self.pipeline.validate_input(user_input)
        lines.append(f"L1 input  : {fmt_checks(report.input_results)}")
        failed = next((r for r in report.input_results if not r.passed), None)
        if failed:
            report.blocked = True
            report.block_reason = f"Input blocked: {failed.category} (confidence={failed.confidence:.2f})"
            return self._finish(report, start, lines, INPUT_BLOCKED_REPLY, user_input, add_to_history=False)

        # L2: NeMo + Claude (its own input rail, dialog rails and output rail).
        answer, nemo_steps, nemo_stopped_by = self.ask_nemo(user_input)
        lines.append(f"L2 NeMo   : {nemo_steps}")
        if nemo_stopped_by:
            report.blocked = True
            # Underscores: the monitor takes the first word after ":" as the category.
            report.block_reason = f"NeMo blocked: nemo_{nemo_stopped_by.replace(' ', '_')}"
            return self._finish(report, start, lines, answer, user_input)

        # L3: output guardrails on what NeMo returned.
        report.output_results, scrubbed = self.pipeline.validate_output(user_input, answer)
        lines.append(f"L3 output : {fmt_checks(report.output_results)}")
        failed = next(
            (r for r in report.output_results
             if not r.passed and r.category != "pii_scrubbing" and r.category not in ADVISORY_CHECKS),
            None,
        )
        if failed:
            report.blocked = True
            report.block_reason = f"Output blocked: {failed.category} (confidence={failed.confidence:.2f})"
            return self._finish(report, start, lines, OUTPUT_BLOCKED_REPLY, user_input)

        leaked = DISCLOSURE_RE.findall(scrubbed)
        if leaked:
            lines.append(f"WARNING   : disclosure terms reached the user: {', '.join(sorted(set(leaked)))}")
        return self._finish(report, start, lines, scrubbed, user_input)

    def _finish(self, report, start, lines, reply, user_input, add_to_history=True):
        report.total_latency_ms = round((time.time() - start) * 1000, 2)
        self.monitor.record(report, user_input)
        if add_to_history:
            # Store what the user actually saw (scrubbed or fallback), never a blocked answer.
            self.history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}]
        verdict = report.block_reason or "delivered"
        lines.append(f"result    : {verdict} in {report.total_latency_ms / 1000:.1f}s")
        return reply, lines

    def run(self):
        print("Harbor Trust Bank's  assistant, full guardrail stack. /trace /reset /stats /quit")
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except EOFError:
                break
            if not user_input or user_input == "/quit":
                break
            if user_input == "/trace":
                self.trace = not self.trace
                print(f"(trace {'on' if self.trace else 'off'})")
                continue
            if user_input == "/reset":
                self.history = []
                print("(new conversation)")
                continue
            if user_input == "/stats":
                self.monitor.print_dashboard()
                continue
            if user_input.startswith("/"):
                print("(commands: /trace /reset /stats /quit)")
                continue

            reply, lines = self.turn(user_input)
            print(f"bot> {reply}")
            if self.trace:
                for line in lines:
                    print(f"     {line}")

        print()
        self.monitor.print_dashboard()


if __name__ == "__main__":
    FullStackChat().run()
