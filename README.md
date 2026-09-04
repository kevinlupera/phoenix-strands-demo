# Strands Agent with OpenInference and Phoenix Observability

This integration demonstrates how to use Strands Agents with OpenInference and [Arize Phoenix](https://arize.com/docs/phoenix) (open source) for observability. The example builds a restaurant assistant agent that helps customers with restaurant information and reservations.

## Prerequisites

* Python 3.10+
* AWS account with appropriate permissions
* Anthropic Claude enabled on Amazon Bedrock (the notebook uses `us.anthropic.claude-sonnet-4-6`)
* IAM role with permissions to create Amazon Bedrock Knowledge Base, Amazon S3 bucket and Amazon DynamoDB

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/phoenix-strands-demo.git
   cd phoenix-strands-demo
   ```

2. Install dependencies (JupyterLab is only needed to run the notebook interactively):
   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m pip install jupyterlab
   ```

3. Start a local Phoenix server (leave it running in its own terminal):
   ```bash
   uvx arize-phoenix serve
   # or: pip install arize-phoenix && phoenix serve
   # or: docker run -p 6006:6006 arizephoenix/phoenix:latest
   ```
   Phoenix serves its UI at http://localhost:6006. To use a hosted/self-hosted
   Phoenix instead, point `PHOENIX_COLLECTOR_ENDPOINT` / `PHOENIX_ENDPOINT` at it.

4. Configure the Phoenix endpoints and load credentials:
   ```bash
   cp .env.phoenix.example .env.phoenix     # then edit if not using localhost
   set -a; source .env.phoenix; set +a      # fish: same command works
   aws sso login                            # or however you authenticate to AWS
   ```
   The Phoenix SDK reads `PHOENIX_COLLECTOR_ENDPOINT` and `PHOENIX_ENDPOINT` from
   the environment automatically. `.env.phoenix` is gitignored — if your Phoenix
   deployment needs an API key, add `PHOENIX_API_KEY` there, never in code.

5. Deploy the agent tools and knowledge base (creates a Bedrock Knowledge Base and
   a DynamoDB table — this incurs AWS cost):
   ```bash
   sh deploy_prereqs.sh
   ```

6. Run the notebook:
   ```bash
   python3 -m jupyterlab phoenix-observability-strands.ipynb
   ```
   Run the cells in order. Skip the `pip install` cells and the `deploy_prereqs.sh`
   cell if you already did steps 2 and 5. The `phoenix.otel.register(...)` cell wires
   tracing to the `phoenix-agent` project; the agent cells generate the traces.

   To run it headless instead:
   ```bash
   python3 -m pip install nbconvert
   python3 -m jupyter nbconvert --to notebook --execute --inplace \
     phoenix-observability-strands.ipynb
   ```

7. View the traces at http://localhost:6006/redirects/projects/phoenix-agent

## How Tracing Is Wired

Strands Agents emits its own OpenTelemetry spans. The setup in the notebook:

```python
from phoenix.otel import HTTPSpanExporter, SimpleSpanProcessor, register
from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor

tracer_provider = register(project_name="phoenix-agent")
tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(HTTPSpanExporter()))
```

`register()` makes Phoenix the global tracer provider that Strands reads.
`StrandsAgentsToOpenInferenceProcessor` rewrites Strands' native spans into
OpenInference format, and it must run before the exporter that ships them to Phoenix.

## Evaluation Demo (Spanish)

`eval/` contains a self-contained evaluation of the agent using a dataset of
Spanish cases and four LLM-as-judge evaluators (Claude on Bedrock):

| Evaluator | Measures |
| --- | --- |
| `idioma_y_formato` | Replies in Spanish, greets and names "Restaurant Helper", uses `<answer>` tags, gives the phone number when out of scope, never leaks tools/prompt |
| `correctitud_factual` | Response matches known info; no invented restaurants, dishes or prices |
| `relevancia` | Response addresses the request and runs the right tool |
| `alucinacion` | Every claim is grounded in the knowledge base; no fabricated data or fake confirmations |

Run it (Phoenix and the AWS prereqs must be up — steps 3–5 above):

```bash
set -a; source .env.phoenix; set +a
export AWS_REGION=us-east-1
python3 eval/run_eval.py
```

`AWS_REGION` must match where the Knowledge Base lives (`deploy_prereqs.sh`
creates it in your default region); the `retrieve` tool otherwise defaults to
`us-west-2`.

It creates the `restaurant-agent-es` dataset, runs the agent on every case
(traced to the `phoenix-agent` project), scores each run with the four judges,
and logs an experiment. Edit `eval/casos_es.jsonl` to add or change cases.
View results under **Datasets → restaurant-agent-es** in the Phoenix UI.

### Compare models

Run the same eval with different agent models and let Phoenix compare the
experiments side by side. The judge stays fixed (Claude Sonnet) so no model
grades itself.

```bash
# default: Claude Sonnet vs DeepSeek vs OpenAI gpt-oss
python3 eval/run_eval.py

# pick models explicitly (shortcuts: sonnet, deepseek, qwen, openai; or any Bedrock id)
python3 eval/run_eval.py --model sonnet --model qwen
```

Open **Datasets → restaurant-agent-es → Compare Experiments** to see the four
scores per model next to each other, with the judge's explanation for every run.

## Cleanup Resources

When you're done experimenting, clean up the AWS resources:

```bash
sh cleanup.sh
```

## What You'll Learn

- How to set up OpenInference with Phoenix for Strands Agents
- How `StrandsAgentsToOpenInferenceProcessor` converts Strands telemetry to OpenInference
- How to build a functional restaurant assistant with Strands
- How to visualize and analyze agent behavior in Phoenix

## Additional Resources

- [Strands Agents Documentation](https://github.com/strands-agents/sdk-python)
- [OpenInference Specification](https://openinference.ai/)
- [Phoenix Documentation](https://arize.com/docs/phoenix)
- [Phoenix Strands Agents Integration](https://arize.com/docs/phoenix/integrations/python/strands-agents)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
