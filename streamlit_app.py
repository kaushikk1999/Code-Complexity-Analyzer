"""Streamlit Community Cloud compatibility entrypoint.

The main application lives in app.py. Keeping this tiny wrapper lets
Streamlit deployments use either `app.py` or the conventional
`streamlit_app.py` file path.
"""

import app  # noqa: F401
