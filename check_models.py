import boto3, os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
client = boto3.client("bedrock", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
models = client.list_foundation_models(byProvider="Anthropic", byOutputModality="TEXT")
print("\nModelos Anthropic en Bedrock:")
print("-" * 60)
for m in models["modelSummaries"]:
    status = m.get("modelLifecycle", {}).get("status", "N/A")
    print(f"  {m['modelId']}  [{status}]")
