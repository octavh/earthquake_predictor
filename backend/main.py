from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Earthquake Forecast API")


@app.get("/")
def read_root():
    return {"status": "alive", "message": "Earthquake forecasting API is running"}


@app.get("/hello", response_class=HTMLResponse)
def hello():
    return """
    <html>
        <head><title>Earthquake Forecast</title></head>
        <body style="font-family: sans-serif; padding: 2rem;">
            <h1>🌍 Earthquake Forecast API</h1>
            <p>Day 1 setup complete. Backend is alive.</p>
            <p>Visit <a href="/docs">/docs</a> for the API documentation.</p>
        </body>
    </html>
    """