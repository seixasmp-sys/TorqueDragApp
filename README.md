# Torque and Drag Sensitivity App v11

Clean rebuilt version.

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Included models

1. Logarithmic soft-string correction
2. Quadratic friction correction
3. Linear decay correction
4. Linear mixture correction

## Main capabilities

- Effective friction coefficient curves
- Normalized drag force curves
- Direct sensitivity to mu1 and mu2
- Sensitivity ratio
- Operational sensitivity to H*, V*, d*
- muT tornado chart
- muT normalized sensitivity
- 2D and 3D response surfaces
- Symbolic endpoint equations in terms of A


## One-click start on Windows

Double-click:

```text
START_APP.bat
```

If port 8501 is already in use, double-click:

```text
START_APP_PORT_8502.bat
```

The command window must stay open while the app is running. Close it or press Ctrl+C to stop the app.


## New in v13

- Replaced top tabs with a highlighted sidebar navigation menu.
- Menu items include icons and larger labels for easier use.


## New in v14

- Fixed duplicate browser tabs when using START_APP.bat.
- The launcher now opens one tab manually and runs Streamlit in headless mode.
