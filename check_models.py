"""List Anthropic models in Amazon Bedrock after explicit confirmation.

This maintenance command always contacts AWS Bedrock. It is intentionally
independent from LLM_PROVIDER and must never run as part of local Ollama tests.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List Anthropic foundation models available in AWS Bedrock."
    )
    parser.add_argument(
        "--confirm-bedrock",
        action="store_true",
        help="Confirm that this command may contact the configured AWS account.",
    )
    args = parser.parse_args()
    if not args.confirm_bedrock:
        parser.error(
            "This command contacts AWS Bedrock. Re-run with --confirm-bedrock "
            "only when that external request is intentional."
        )

    import boto3

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    client = boto3.client(
        "bedrock",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    models = client.list_foundation_models(
        byProvider="Anthropic",
        byOutputModality="TEXT",
    )
    print("\nAnthropic models in Bedrock:")
    print("-" * 60)
    for model in models["modelSummaries"]:
        status = model.get("modelLifecycle", {}).get("status", "N/A")
        print(f"  {model['modelId']}  [{status}]")


if __name__ == "__main__":
    main()
