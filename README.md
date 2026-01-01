# Speculation Stock Agent (Streamlit Prototype + LLM Thesis)

This Streamlit app:
- Scans a list of tickers
- Computes momentum / volatility / volume-spike / RSI signals
- Ranks by a transparent **Speculation Score**
- Suggests a monthly allocation (e.g., $100/month)
- Optionally generates **Thesis + Invalidation rules** using the OpenAI API (LLM-assisted)

> Educational use only. Not financial advice.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Enable LLM-assisted thesis

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

Restart Streamlit, then check **Enable LLM-assisted thesis** in the sidebar.

## API key safety
Do not hardcode keys in code. Use environment variables or your deployment provider’s secrets manager.
