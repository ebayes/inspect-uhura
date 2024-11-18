#### Development

To work on development of Inspect, clone the repository and install with the `-e` flag and `[dev]` optional dependencies:

```         
$ git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
$ cd inspect_ai
$ pip install -e ".[dev]"
$ pip install google-generativeai
$ export GOOGLE_API_KEY=your-google-api-key
$ python 0_arc_easy.py
```

If you use VS Code, you should be sure to have installed the recommended extensions (Python, Ruff, and MyPy). Note that you'll be prompted to install these when you open the project in VS Code.
