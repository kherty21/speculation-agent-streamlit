# Speculation Stock Agent

Adds:
- Decision logging to `decision_journal.csv`
- Performance tracking per decision vs benchmark (QQQ)
- Download/Upload journal controls (useful for Streamlit Cloud where disk can be ephemeral)

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## OpenAI key (for thesis)
```bash
export OPENAI_API_KEY="sk-..."
```
Restart Streamlit after setting the key.
