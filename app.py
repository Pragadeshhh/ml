"""
Root entry point for Streamlit application.
Allows running 'streamlit run app.py' directly from the root workspace.
"""
import os
import runpy

app_path = os.path.join(os.path.dirname(__file__), 'app', 'app.py')
runpy.run_path(app_path, run_name='__main__')
